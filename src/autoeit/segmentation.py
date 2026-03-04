"""Audio segmentation module.

The most critical module — extracts exactly 30 participant response segments
from each EIT recording, separating them from stimulus playback and silence.

EIT structure (per item):
  Native stimulus (~3-8s) → Tone beep (~0.5s) → Participant response (~2-10s) → Silence (~5-10s)

Strategy (hybrid):
  1. Detect tones (reliable ~1kHz beeps between stimulus and response)
  2. Detect non-silent regions
  3. Classify each speech region as stimulus or response based on position relative to tones
  4. Validate: expect exactly 30 response segments
  5. Fallback: use Whisper full-file pass to match known stimuli and infer response boundaries
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

    This is the main entry point. It tries multiple strategies in order
    of reliability and falls back as needed.

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
        Exactly 30 response segments (or best effort if validation fails).
    """
    logger.info("Starting segmentation (%.1fs audio @ %dHz)", len(audio) / sr, sr)

    # ------------------------------------------------------------------
    # Step 1: Detect tones
    # ------------------------------------------------------------------
    tones = detect_tones(audio, sr, config)
    logger.info("Detected %d tone events", len(tones))

    # ------------------------------------------------------------------
    # Step 2: Detect all non-silent regions
    # ------------------------------------------------------------------
    speech_regions = detect_non_silent(audio, sr, config)
    logger.info("Detected %d non-silent regions", len(speech_regions))

    # ------------------------------------------------------------------
    # Step 3: Try tone-based segmentation (most reliable)
    # ------------------------------------------------------------------
    if len(tones) >= config.expected_segments * 0.7:  # at least 70% of expected tones
        logger.info("Using tone-based segmentation (%d tones found)", len(tones))
        responses = extract_responses_from_tones(audio, sr, tones, speech_regions, config)
        if _validate_segments(responses, config):
            return responses
        logger.warning(
            "Tone-based segmentation produced %d segments (expected %d) — trying hybrid",
            len(responses), config.expected_segments,
        )

    # ------------------------------------------------------------------
    # Step 4: Hybrid — use speech regions + pattern matching
    # ------------------------------------------------------------------
    logger.info("Using hybrid segmentation (speech regions + pattern analysis)")
    responses = extract_responses_hybrid(audio, sr, speech_regions, tones, config)
    if _validate_segments(responses, config):
        return responses

    logger.warning(
        "Hybrid segmentation produced %d segments (expected %d) — trying silence-only",
        len(responses), config.expected_segments,
    )

    # ------------------------------------------------------------------
    # Step 5: Fallback — pure silence-based with alternating heuristic
    # ------------------------------------------------------------------
    logger.info("Falling back to silence-based alternating segmentation")
    responses = extract_responses_silence_alternating(audio, sr, speech_regions, config)

    if len(responses) != config.expected_segments:
        logger.warning(
            "Final segmentation: %d segments (expected %d). Results may need manual review.",
            len(responses), config.expected_segments,
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

    Returns list of (start_sample, end_sample) for each detected tone.
    """
    from scipy import signal as scipy_signal

    target_freq = config.tone_freq_hz
    min_duration = config.tone_min_duration_ms / 1000.0
    max_duration = config.tone_max_duration_ms / 1000.0

    # Short-time energy in the tone frequency band
    # Bandpass filter around the tone frequency (±200 Hz)
    low_freq = max(target_freq - 200, 100)
    high_freq = min(target_freq + 200, sr / 2 - 1)

    # Design bandpass filter
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

    # Threshold: tone regions have high energy in the band
    threshold = np.percentile(envelope, 95) * 0.3  # adaptive threshold
    if threshold < 0.01:
        threshold = 0.01

    # Find contiguous regions above threshold
    above = envelope > threshold
    tone_regions = _contiguous_regions(above)

    # Filter by duration
    tones = []
    for start, end in tone_regions:
        duration = (end - start) / sr
        if min_duration <= duration <= max_duration:
            # Verify it's actually tonal (high spectral purity)
            segment = audio[start:end]
            if _is_tonal(segment, sr, target_freq):
                tones.append((start, end))

    return tones


def _is_tonal(segment: np.ndarray, sr: int, expected_freq: float, tolerance: float = 300.0) -> bool:
    """Check if a segment is a pure tone near the expected frequency."""
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

    return (peak_energy / total_energy) > 0.2


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
# Response extraction strategies
# ---------------------------------------------------------------------------

def extract_responses_from_tones(
    audio: np.ndarray,
    sr: int,
    tones: List[Tuple[int, int]],
    speech_regions: List[Tuple[int, int]],
    config: SegmentationConfig,
) -> List[AudioSegment]:
    """Extract responses using tone positions as anchors.

    After each tone, the first speech region is the participant response.
    """
    responses = []
    padding_before = int(config.padding_before_ms * sr / 1000)
    padding_after = int(config.padding_after_ms * sr / 1000)
    max_response = int(config.max_response_duration_ms * sr / 1000)
    min_response = int(config.min_response_duration_ms * sr / 1000)

    for idx, (tone_start, tone_end) in enumerate(tones):
        # Look for the first speech region after this tone
        best_region = None
        for reg_start, reg_end in speech_regions:
            # Region must start after the tone ends
            if reg_start > tone_end:
                # But not too far away (within 15 seconds)
                gap_s = (reg_start - tone_end) / sr
                if gap_s > 15.0:
                    break  # too far, probably next item's stimulus
                best_region = (reg_start, reg_end)
                break

        if best_region is None:
            # No speech after tone — participant didn't respond
            responses.append(AudioSegment(
                start_sample=tone_end,
                end_sample=min(tone_end + sr, len(audio)),
                start_s=tone_end / sr,
                end_s=min(tone_end + sr, len(audio)) / sr,
                segment_type="response",
                sentence_number=idx + 1,
                audio=np.zeros(sr, dtype=np.float32),
                energy_rms=0.0,
                notes="no_response",
            ))
            continue

        reg_start, reg_end = best_region

        # Apply padding
        start = max(0, reg_start - padding_before)
        end = min(len(audio), reg_end + padding_after)

        # Cap duration
        if end - start > max_response:
            end = start + max_response

        segment_audio = audio[start:end].copy()
        rms = float(np.sqrt(np.mean(segment_audio ** 2))) if len(segment_audio) > 0 else 0.0

        responses.append(AudioSegment(
            start_sample=start,
            end_sample=end,
            start_s=start / sr,
            end_s=end / sr,
            segment_type="response",
            sentence_number=idx + 1,
            audio=segment_audio,
            energy_rms=rms,
        ))

    return responses[:config.expected_segments]


def extract_responses_hybrid(
    audio: np.ndarray,
    sr: int,
    speech_regions: List[Tuple[int, int]],
    tones: List[Tuple[int, int]],
    config: SegmentationConfig,
) -> List[AudioSegment]:
    """Hybrid segmentation using speech regions and optional tone info.

    The EIT has a repeating pattern: stimulus → tone → response → silence.
    Speech regions alternate: stimulus, response, stimulus, response, ...

    We use energy/duration characteristics to distinguish:
    - Stimuli: consistent energy (pre-recorded native speaker)
    - Responses: variable energy (non-native participant)
    """
    if len(speech_regions) < 2:
        return []

    padding_before = int(config.padding_before_ms * sr / 1000)
    padding_after = int(config.padding_after_ms * sr / 1000)
    max_response = int(config.max_response_duration_ms * sr / 1000)

    # Compute features for each speech region
    features = []
    for start, end in speech_regions:
        seg = audio[start:end]
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0
        duration = (end - start) / sr

        # Check if a tone precedes this region (within 5s)
        preceded_by_tone = False
        for tone_start, tone_end in tones:
            gap = (start - tone_end) / sr
            if 0 < gap < 5.0:
                preceded_by_tone = True
                break

        features.append({
            "start": start,
            "end": end,
            "rms": rms,
            "duration": duration,
            "preceded_by_tone": preceded_by_tone,
        })

    # Strategy: regions preceded by a tone are responses
    # Remaining regions are paired: odd-indexed = stimulus, even = response
    # within each stimulus-response pair

    responses = []
    used = set()

    # First pass: tone-preceded regions
    for i, feat in enumerate(features):
        if feat["preceded_by_tone"]:
            responses.append(_make_segment(audio, sr, feat, len(responses) + 1, padding_before, padding_after, max_response))
            used.add(i)

    # If we got enough from tones alone
    if len(responses) >= config.expected_segments:
        return responses[:config.expected_segments]

    # Second pass: alternate remaining regions
    remaining = [(i, f) for i, f in enumerate(features) if i not in used]

    if len(remaining) >= 2:
        # Group remaining into pairs (stimulus, response)
        # The pattern after intro: stimulus → response → stimulus → response → ...
        # First remaining region is likely a stimulus
        for pair_idx in range(0, len(remaining) - 1, 2):
            if len(responses) >= config.expected_segments:
                break
            # Second of each pair is the response
            _, resp_feat = remaining[pair_idx + 1]
            responses.append(_make_segment(
                audio, sr, resp_feat, len(responses) + 1,
                padding_before, padding_after, max_response,
            ))
    elif len(remaining) == 1 and len(responses) < config.expected_segments:
        # Single remaining — might be a response
        _, feat = remaining[0]
        responses.append(_make_segment(
            audio, sr, feat, len(responses) + 1,
            padding_before, padding_after, max_response,
        ))

    return responses[:config.expected_segments]


def extract_responses_silence_alternating(
    audio: np.ndarray,
    sr: int,
    speech_regions: List[Tuple[int, int]],
    config: SegmentationConfig,
) -> List[AudioSegment]:
    """Fallback: treat every other speech region as a response.

    Assumes the pattern: stimulus(1), response(1), stimulus(2), response(2), ...
    So indices 1, 3, 5, ... are responses.
    """
    padding_before = int(config.padding_before_ms * sr / 1000)
    padding_after = int(config.padding_after_ms * sr / 1000)
    max_response = int(config.max_response_duration_ms * sr / 1000)

    responses = []
    # Try: response at odd indices (0-based: 1, 3, 5, ...)
    for i in range(1, len(speech_regions), 2):
        if len(responses) >= config.expected_segments:
            break
        start, end = speech_regions[i]
        seg = audio[start:end]
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0
        feat = {"start": start, "end": end, "rms": rms, "duration": (end - start) / sr}
        responses.append(_make_segment(
            audio, sr, feat, len(responses) + 1,
            padding_before, padding_after, max_response,
        ))

    return responses[:config.expected_segments]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(
    audio: np.ndarray,
    sr: int,
    feat: Dict[str, Any],
    sentence_num: int,
    padding_before: int,
    padding_after: int,
    max_response: int,
) -> AudioSegment:
    """Build an AudioSegment from feature dict."""
    start = max(0, feat["start"] - padding_before)
    end = min(len(audio), feat["end"] + padding_after)
    if end - start > max_response:
        end = start + max_response

    seg_audio = audio[start:end].copy()
    rms = float(np.sqrt(np.mean(seg_audio ** 2))) if len(seg_audio) > 0 else 0.0

    return AudioSegment(
        start_sample=start,
        end_sample=end,
        start_s=start / sr,
        end_s=end / sr,
        segment_type="response",
        sentence_number=sentence_num,
        audio=seg_audio,
        energy_rms=rms,
    )


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


def _validate_segments(
    segments: List[AudioSegment],
    config: SegmentationConfig,
) -> bool:
    """Check if segmentation produced the expected number of segments."""
    n = len(segments)
    expected = config.expected_segments
    # Accept within ±2 of expected
    if abs(n - expected) <= 2:
        if n != expected:
            logger.warning("Got %d segments, expected %d (within tolerance)", n, expected)
        return True
    return False
