"""Audio segmentation module.

The most critical module — extracts exactly 30 participant response segments
from each EIT recording, separating them from stimulus playback and silence.

EIT structure (per item):
  Native stimulus (~3-8s) → Tone beep (~0.5s) → Participant response (~2-10s) → Silence (~5-10s)

Strategy:
  1. Detect non-silent speech regions in the audio
  2. Detect tone beeps (~1kHz) that mark the boundary between stimulus and response
  3. Group nearby speech regions into "items" separated by long silence gaps
  4. Within each item, split into stimulus portion (before tone) and response portion (after tone)
  5. Sort all response segments chronologically and number them 1-N
  6. Validate against expected count (30)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import SegmentationConfig, TARGET_SENTENCES

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """A detected audio segment."""
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float
    segment_type: str = "unknown"  # "stimulus", "response", "tone", "silence"
    sentence_number: int = 0       # 1-30 if mapped
    audio: Optional[np.ndarray] = field(default=None, repr=False)
    energy_rms: float = 0.0
    notes: str = ""


def segment_audio(
    audio: np.ndarray,
    sr: int,
    config: SegmentationConfig,
) -> List[AudioSegment]:
    """Extract 30 participant response segments from preprocessed audio.

    This is the main entry point. It uses a group-based approach:
    1. Detect all non-silent speech regions
    2. Detect tone beeps
    3. Group nearby regions into EIT "items" (separated by silence gaps)
    4. Within each item, extract the response portion (after the tone)
    5. Sort chronologically, number 1-N

    Parameters
    ----------
    audio : np.ndarray
        Preprocessed audio (float32, mono, after intro skip).
    sr : int
        Sample rate.
    config : SegmentationConfig
        Segmentation parameters.

    Returns
    -------
    List[AudioSegment]
        Response segments sorted chronologically, numbered 1-N.
    """
    total_duration = len(audio) / sr
    logger.info("Starting segmentation (%.1fs audio @ %dHz)", total_duration, sr)

    # ------------------------------------------------------------------
    # Step 1: Detect tone beeps
    # ------------------------------------------------------------------
    tones = detect_tones(audio, sr, config)
    logger.info("Detected %d tone events", len(tones))

    # ------------------------------------------------------------------
    # Step 2: Detect all non-silent regions
    # ------------------------------------------------------------------
    speech_regions = detect_non_silent(audio, sr, config)
    logger.info("Detected %d non-silent regions", len(speech_regions))

    if len(speech_regions) == 0:
        logger.warning("No non-silent regions detected!")
        return []

    # ------------------------------------------------------------------
    # Step 3: Group nearby regions into EIT "items"
    # ------------------------------------------------------------------
    items = _group_into_items(speech_regions, sr, config, tones)
    logger.info("Grouped into %d EIT items", len(items))

    # ------------------------------------------------------------------
    # Step 4: Extract response from each item
    # ------------------------------------------------------------------
    responses = _extract_responses_from_items(audio, sr, items, tones, config)
    logger.info("Extracted %d response segments", len(responses))

    # ------------------------------------------------------------------
    # Step 5: Sort chronologically and assign sentence numbers
    # ------------------------------------------------------------------
    responses.sort(key=lambda s: s.start_s)
    for i, seg in enumerate(responses):
        seg.sentence_number = i + 1

    if len(responses) != config.expected_segments:
        logger.warning(
            "Got %d segments, expected %d%s",
            len(responses),
            config.expected_segments,
            " (within tolerance)" if abs(len(responses) - config.expected_segments) <= 2 else "",
        )

    return responses


# ---------------------------------------------------------------------------
# Tone detection
# ---------------------------------------------------------------------------

def detect_tones(
    audio: np.ndarray,
    sr: int,
    config: SegmentationConfig,
) -> List[Tuple[int, int]]:
    """Detect tone beeps (~1kHz) in the audio.

    Uses bandpass filtering + envelope detection with relaxed thresholds
    for better recall. Returns list of (start_sample, end_sample).
    """
    from scipy import signal as scipy_signal

    target_freq = config.tone_freq_hz
    min_duration = config.tone_min_duration_ms / 1000.0
    max_duration = config.tone_max_duration_ms / 1000.0

    # Bandpass filter around the tone frequency (±300 Hz for wider capture)
    low_freq = max(target_freq - 300, 100)
    high_freq = min(target_freq + 300, sr / 2 - 1)

    nyq = sr / 2.0
    b, a = scipy_signal.butter(
        4,
        [low_freq / nyq, high_freq / nyq],
        btype="band",
    )
    filtered = scipy_signal.filtfilt(b, a, audio)

    # Compute envelope via Hilbert transform
    analytic = scipy_signal.hilbert(filtered)
    envelope = np.abs(analytic)

    # Smooth the envelope
    win_size = int(0.02 * sr)  # 20ms window
    if win_size > 0:
        kernel = np.ones(win_size) / win_size
        envelope = np.convolve(envelope, kernel, mode="same")

    # Adaptive threshold — use a lower percentile for better recall
    threshold = np.percentile(envelope, 90) * 0.25
    threshold = max(threshold, 0.005)

    # Find contiguous regions above threshold
    above = envelope > threshold
    tone_regions = _contiguous_regions(above)

    # Filter by duration and tonal quality
    tones = []
    for start, end in tone_regions:
        duration = (end - start) / sr
        if min_duration <= duration <= max_duration:
            segment = audio[start:end]
            if _is_tonal(segment, sr, target_freq):
                tones.append((start, end))

    # Merge tones that are very close together (within 100ms) — sometimes
    # a single beep gets split by envelope dips
    tones = _merge_close_regions(tones, min_gap=int(0.1 * sr))

    return tones


def _is_tonal(segment: np.ndarray, sr: int, expected_freq: float, tolerance: float = 400.0) -> bool:
    """Check if a segment is a pure tone near the expected frequency.

    Uses a relaxed tolerance (400Hz) and lower spectral purity threshold (0.1)
    to improve recall — we'd rather detect a few false tones and filter later
    than miss real tones.
    """
    if len(segment) < 256:
        return False

    # FFT
    n = len(segment)
    fft_vals = np.abs(np.fft.rfft(segment * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # Find peak frequency
    peak_idx = np.argmax(fft_vals)
    peak_freq = freqs[peak_idx]

    # Check if peak is near expected tone
    if abs(peak_freq - expected_freq) > tolerance:
        return False

    # Check spectral purity: peak should contain significant energy
    total_energy = np.sum(fft_vals ** 2)
    if total_energy < 1e-10:
        return False

    # Energy in ±100Hz around peak
    mask = (freqs >= peak_freq - 100) & (freqs <= peak_freq + 100)
    peak_energy = np.sum(fft_vals[mask] ** 2)

    return (peak_energy / total_energy) > 0.1


# ---------------------------------------------------------------------------
# Silence / non-silent detection
# ---------------------------------------------------------------------------

def detect_non_silent(
    audio: np.ndarray,
    sr: int,
    config: SegmentationConfig,
) -> List[Tuple[int, int]]:
    """Detect non-silent regions using energy-based approach.

    Returns list of (start_sample, end_sample).
    """
    # Convert parameters
    min_silence_samples = int(config.min_silence_len_ms * sr / 1000)
    frame_size = int(config.seek_step_ms * sr / 1000)

    if frame_size < 1:
        frame_size = int(0.01 * sr)

    # Compute frame-wise RMS energy
    n_frames = len(audio) // frame_size
    if n_frames == 0:
        return []

    rms_frames = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * frame_size: (i + 1) * frame_size]
        rms_frames[i] = np.sqrt(np.mean(frame ** 2))

    # Convert silence threshold from dBFS to linear
    # dBFS reference: 1.0 = 0dBFS for float audio
    silence_thresh_linear = 10 ** (config.silence_thresh_db / 20.0)

    # Find frames above threshold
    is_speech = rms_frames > silence_thresh_linear

    # Merge nearby speech frames (bridge small silence gaps)
    min_silence_frames = min_silence_samples // frame_size
    speech_regions_frames = _merge_close_regions(
        _contiguous_regions(is_speech),
        min_gap=min_silence_frames // 2,
    )

    # Convert back to samples
    regions = []
    for start_frame, end_frame in speech_regions_frames:
        start_sample = start_frame * frame_size
        end_sample = min(end_frame * frame_size, len(audio))
        duration_ms = (end_sample - start_sample) * 1000 / sr
        if duration_ms >= config.min_response_duration_ms / 2:  # low threshold here
            regions.append((start_sample, end_sample))

    return regions


# ---------------------------------------------------------------------------
# Item grouping and response extraction
# ---------------------------------------------------------------------------

@dataclass
class _EITItem:
    """An EIT item: one stimulus-tone-response cycle."""
    regions: List[Tuple[int, int]]   # non-silent regions in this item
    tone: Optional[Tuple[int, int]]  # tone beep if detected, else None
    item_start: int                  # earliest sample
    item_end: int                    # latest sample


def _group_into_items(
    speech_regions: List[Tuple[int, int]],
    sr: int,
    config: SegmentationConfig,
    tones: List[Tuple[int, int]],
) -> List[_EITItem]:
    """Group speech regions into EIT items based on silence gaps.

    Each EIT item (stimulus+tone+response) is separated from the next by
    a substantial silence gap (typically 2-6s). Within an item, the
    stimulus, tone, and response may be close together or even merged
    into one region.

    We use an adaptive gap threshold: compute gaps between consecutive
    non-silent regions and look for the natural break points.
    """

    if len(speech_regions) < 2:
        return [_EITItem(
            regions=speech_regions,
            tone=None,
            item_start=speech_regions[0][0] if speech_regions else 0,
            item_end=speech_regions[0][1] if speech_regions else 0,
        )]

    # ---- Compute gaps between consecutive regions ----
    gaps = []
    for i in range(len(speech_regions) - 1):
        gap_start = speech_regions[i][1]
        gap_end = speech_regions[i + 1][0]
        gap_s = (gap_end - gap_start) / sr
        gaps.append(gap_s)

    # ---- Find adaptive gap threshold ----
    # We expect ~30 items over the audio, so ~29 big gaps.
    # Sort gaps and look for a natural break.
    sorted_gaps = sorted(gaps, reverse=True)

    # Target: ~expected_segments items, so ~(expected_segments - 1) splits
    target_splits = config.expected_segments - 1

    if len(sorted_gaps) > target_splits:
        # Use the gap between the target_splits-th and (target_splits+1)-th
        # largest gaps as the threshold
        gap_threshold_s = (sorted_gaps[target_splits - 1] + sorted_gaps[target_splits]) / 2.0
        # But enforce a minimum (at least 2s gap to be an item boundary)
        gap_threshold_s = max(gap_threshold_s, 2.0)
    else:
        # Fewer gaps than expected — use a fixed threshold
        gap_threshold_s = 2.0

    logger.debug("Gap threshold: %.2fs (gaps range: %.2f - %.2fs)",
                 gap_threshold_s, min(gaps), max(gaps))

    # ---- Group regions by gap threshold ----
    groups: List[List[Tuple[int, int]]] = [[speech_regions[0]]]
    for i, gap in enumerate(gaps):
        if gap >= gap_threshold_s:
            groups.append([speech_regions[i + 1]])
        else:
            groups[-1].append(speech_regions[i + 1])

    # ---- Build EITItem objects, attaching tones ----
    items: List[_EITItem] = []
    for group in groups:
        item_start = group[0][0]
        item_end = group[-1][1]

        # Find tone within this item's time span (with small margin)
        margin = int(0.5 * sr)
        item_tone = None
        for tone_start, tone_end in tones:
            if tone_start >= item_start - margin and tone_end <= item_end + margin:
                item_tone = (tone_start, tone_end)
                break

        items.append(_EITItem(
            regions=group,
            tone=item_tone,
            item_start=item_start,
            item_end=item_end,
        ))

    return items


def _extract_responses_from_items(
    audio: np.ndarray,
    sr: int,
    items: List[_EITItem],
    tones: List[Tuple[int, int]],
    config: SegmentationConfig,
) -> List[AudioSegment]:
    """Extract the response portion from each EIT item.

    For each item:
    - If a tone was detected: response = speech AFTER the tone
    - If no tone detected but multiple regions: response = last region
      (stimulus is first, response is last)
    - If only one region: treat entirety as potential response (might be
      stimulus-only if participant didn't respond, or might be a merged
      stimulus+response)

    For single-region items, we try to split on the tone frequency or
    use duration heuristics.
    """
    padding_before = int(config.padding_before_ms * sr / 1000)
    padding_after = int(config.padding_after_ms * sr / 1000)
    max_response = int(config.max_response_duration_ms * sr / 1000)
    min_response_samples = int(config.min_response_duration_ms * sr / 1000)

    responses: List[AudioSegment] = []

    for item in items:
        resp_start = None
        resp_end = None
        notes = ""

        if item.tone is not None:
            # -- TONE DETECTED: response is speech after the tone --
            tone_end_sample = item.tone[1]

            # Find speech regions within this item that start AFTER the tone
            post_tone_regions = [
                (s, e) for s, e in item.regions
                if s >= tone_end_sample - int(0.05 * sr)  # small tolerance
            ]

            if post_tone_regions:
                # Merge all post-tone regions into one response
                resp_start = post_tone_regions[0][0]
                resp_end = post_tone_regions[-1][1]
            else:
                # No speech after tone → no response
                resp_start = tone_end_sample
                resp_end = min(tone_end_sample + int(0.5 * sr), len(audio))
                notes = "no_response"

        elif len(item.regions) >= 2:
            # -- NO TONE, MULTIPLE REGIONS: last region is likely the response --
            # (First region = stimulus, last = response)
            last_region = item.regions[-1]
            resp_start = last_region[0]
            resp_end = last_region[1]

        else:
            # -- SINGLE REGION, NO TONE --
            # Could be: (a) stimulus only (no response), (b) merged stim+response
            # Try to split: look for a tone-like event within the region
            region_start, region_end = item.regions[0]
            region_duration = (region_end - region_start) / sr

            if region_duration < 2.0:
                # Very short — likely just stimulus or just response
                # Use the whole region as a potential response
                resp_start = region_start
                resp_end = region_end
            else:
                # Longer region — try to find a tonal boundary within it
                # to split stimulus from response
                split_point = _find_tone_boundary_in_region(
                    audio, sr, region_start, region_end, config
                )
                if split_point is not None:
                    resp_start = split_point
                    resp_end = region_end
                else:
                    # No boundary found — use second half as a heuristic
                    # (stimulus first, response second)
                    midpoint = (region_start + region_end) // 2
                    resp_start = midpoint
                    resp_end = region_end

        if resp_start is None:
            continue

        # Apply padding
        start = max(0, resp_start - padding_before)
        end = min(len(audio), resp_end + padding_after)

        # Cap duration
        if end - start > max_response:
            end = start + max_response

        seg_audio = audio[start:end].copy()
        rms = float(np.sqrt(np.mean(seg_audio ** 2))) if len(seg_audio) > 0 else 0.0

        # Check if this is essentially silence (no actual response)
        if rms < 0.005 and notes != "no_response":
            notes = "no_response"

        responses.append(AudioSegment(
            start_sample=start,
            end_sample=end,
            start_s=start / sr,
            end_s=end / sr,
            segment_type="response",
            sentence_number=0,  # will be set after sorting
            audio=seg_audio,
            energy_rms=rms,
            notes=notes,
        ))

    return responses


def _find_tone_boundary_in_region(
    audio: np.ndarray,
    sr: int,
    region_start: int,
    region_end: int,
    config: SegmentationConfig,
) -> Optional[int]:
    """Look for a tone-like boundary within a single speech region.

    Returns the sample position just after the tone, or None if not found.
    """
    from scipy import signal as scipy_signal

    segment = audio[region_start:region_end]
    if len(segment) < int(0.1 * sr):
        return None

    target_freq = config.tone_freq_hz
    low_freq = max(target_freq - 300, 100)
    high_freq = min(target_freq + 300, sr / 2 - 1)

    nyq = sr / 2.0
    b, a = scipy_signal.butter(4, [low_freq / nyq, high_freq / nyq], btype="band")
    filtered = scipy_signal.filtfilt(b, a, segment)
    envelope = np.abs(scipy_signal.hilbert(filtered))

    # Smooth
    win_size = int(0.02 * sr)
    if win_size > 0:
        kernel = np.ones(win_size) / win_size
        envelope = np.convolve(envelope, kernel, mode="same")

    # Look for a peak in the tonal envelope
    threshold = np.percentile(envelope, 80) * 0.5
    if threshold < 0.003:
        return None

    above = envelope > threshold
    tone_candidates = _contiguous_regions(above)

    min_dur = config.tone_min_duration_ms / 1000.0
    max_dur = config.tone_max_duration_ms / 1000.0

    for start, end in tone_candidates:
        dur = (end - start) / sr
        if min_dur * 0.5 <= dur <= max_dur:
            # Found a tonal region — response starts after it
            return region_start + end

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous True regions in a boolean array.

    Returns list of (start_idx, end_idx) pairs.
    """
    if len(mask) == 0:
        return []

    regions = []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [len(mask)]))

    for s, e in zip(starts, ends):
        regions.append((int(s), int(e)))

    return regions


def _merge_close_regions(
    regions: List[Tuple[int, int]],
    min_gap: int,
) -> List[Tuple[int, int]]:
    """Merge regions that are separated by less than *min_gap*."""
    if not regions:
        return []

    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= min_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged
