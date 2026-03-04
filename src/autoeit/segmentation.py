"""Audio segmentation module — Phase 1 (Advanced).

Extracts exactly 30 participant response segments from each EIT recording
using three complementary signals fused together:

1. **Silero VAD** — Neural voice activity detection (replaces energy-based).
   Robust to noise, music, and varying recording conditions.

2. **Tone detection** — Bandpass filter + envelope + spectral purity to
   locate ~1 kHz beeps that separate each stimulus from its response.

3. **Full-file ASR text matching** (optional) — A word-timestamped
   transcript of the full audio is fuzzy-matched against the 30 known
   stimulus sentences to locate stimulus playback times precisely.

Fusion strategy:
  · With ASR word timeline AND >=20 matched stimuli:
        Use matched stimulus positions + tone anchors + VAD.
  · Without word timeline OR <20 matches:
        Fall back to VAD + tone adaptive-gap grouping.

EIT item structure:
  Native stimulus -> Tone beep -> Participant response -> Silence -> ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import SegmentationConfig, TARGET_SENTENCES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level Silero VAD model cache (loaded once, reused across calls)
# ---------------------------------------------------------------------------
_vad_model = None
_vad_utils = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AudioSegment:
    """A detected audio segment."""
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float
    segment_type: str = "unknown"      # "stimulus" | "response" | "tone" | "silence"
    sentence_number: int = 0           # 1-30 if mapped
    audio: Optional[np.ndarray] = field(default=None, repr=False)
    energy_rms: float = 0.0
    notes: str = ""
    pre_transcription: str = ""        # response text from full-file ASR (if available)


# ======================================================================
#  MAIN ENTRY POINT
# ======================================================================

def segment_audio(
    audio: np.ndarray,
    sr: int,
    config: SegmentationConfig,
    *,
    word_timeline: Optional[List[Dict[str, Any]]] = None,
) -> List[AudioSegment]:
    """Extract participant response segments from preprocessed audio.

    Parameters
    ----------
    audio : np.ndarray
        Preprocessed audio (float32, mono, after intro skip).
    sr : int
        Sample rate (should be 16000 for Silero VAD).
    config : SegmentationConfig
        Segmentation parameters.
    word_timeline : list of dict, optional
        Word-level timestamps from full-file ASR transcription.
        Each dict: ``{"word": str, "start": float, "end": float}``.
        When provided, enables text-matching strategy.

    Returns
    -------
    List[AudioSegment]
        Response segments sorted chronologically, numbered 1-N.
    """
    total_duration = len(audio) / sr
    logger.info("Phase 1 segmentation — %.1fs audio @ %d Hz", total_duration, sr)

    # ---- Signal 1: Silero VAD ----
    vad_regions = _silero_vad_detect(audio, sr)
    logger.info("  Silero VAD  -> %d speech regions", len(vad_regions))

    # ---- Signal 2: Tone detection ----
    tones = detect_tones(audio, sr, config)
    logger.info("  Tones       -> %d detected", len(tones))

    # ---- Signal 3: Text matching (optional) ----
    stim_matches: List[Dict[str, Any]] = []
    if word_timeline and len(word_timeline) > 0:
        stim_matches = _match_stimuli_in_transcript(
            word_timeline, TARGET_SENTENCES,
        )
        logger.info(
            "  Text match  -> %d / %d stimuli located",
            len(stim_matches), len(TARGET_SENTENCES),
        )

    # ---- Segmentation: always use VAD + TONE GROUPING ----
    # (Text matching is unreliable for EIT because the mic mainly captures
    #  participant responses, not native-speaker stimulus playback. Whisper
    #  transcribes responses, so matching them as stimuli mislocates
    #  boundaries. VAD + tone grouping is more robust.)
    logger.info("Strategy: VAD + TONE GROUPING")
    responses = _extract_responses_vad_tones(
        audio, sr, vad_regions, tones, config,
    )

    # ---- Final ordering ----
    responses.sort(key=lambda s: s.start_s)
    for i, seg in enumerate(responses):
        seg.sentence_number = i + 1

    # ---- Enrich with pre-transcription from full-file ASR ----
    if word_timeline:
        _attach_pre_transcriptions(responses, word_timeline)
        n_enriched = sum(1 for s in responses if s.pre_transcription)
        logger.info("  Pre-transcriptions attached: %d / %d", n_enriched, len(responses))

    if len(responses) != config.expected_segments:
        logger.warning(
            "Got %d segments, expected %d",
            len(responses), config.expected_segments,
        )

    return responses


# ======================================================================
#  SIGNAL 1: SILERO VAD
# ======================================================================

def _load_silero_vad():
    """Load (or return cached) Silero VAD model."""
    global _vad_model, _vad_utils
    if _vad_model is not None:
        return _vad_model, _vad_utils

    import torch
    logger.info("Loading Silero VAD model ...")
    _vad_model, _vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    logger.info("Silero VAD model loaded")
    return _vad_model, _vad_utils


def _silero_vad_detect(
    audio: np.ndarray,
    sr: int,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
    threshold: float = 0.35,
) -> List[Tuple[int, int]]:
    """Detect speech regions via Silero VAD.

    Returns list of ``(start_sample, end_sample)`` pairs.
    """
    import torch

    model, utils = _load_silero_vad()
    get_speech_timestamps = utils[0]

    tensor = torch.from_numpy(audio.astype(np.float32))
    ts = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=sr,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        threshold=threshold,
    )
    model.reset_states()
    return [(t["start"], t["end"]) for t in ts]


def detect_non_silent(
    audio: np.ndarray,
    sr: int,
    config: SegmentationConfig,
) -> List[Tuple[int, int]]:
    """Backward-compatible wrapper — now delegates to Silero VAD."""
    return _silero_vad_detect(audio, sr)


# ======================================================================
#  SIGNAL 2: TONE DETECTION
# ======================================================================

def detect_tones(
    audio: np.ndarray,
    sr: int,
    config: SegmentationConfig,
) -> List[Tuple[int, int]]:
    """Detect ~1 kHz tone beeps via bandpass + envelope + spectral purity.

    Returns list of ``(start_sample, end_sample)`` pairs.
    """
    from scipy import signal as scipy_signal

    target_freq = config.tone_freq_hz
    min_dur = config.tone_min_duration_ms / 1000.0
    max_dur = config.tone_max_duration_ms / 1000.0

    low = max(target_freq - 300, 100)
    high = min(target_freq + 300, sr / 2 - 1)
    nyq = sr / 2.0
    b, a = scipy_signal.butter(4, [low / nyq, high / nyq], btype="band")
    filtered = scipy_signal.filtfilt(b, a, audio)

    envelope = np.abs(scipy_signal.hilbert(filtered))
    win = int(0.02 * sr)
    if win > 0:
        kernel = np.ones(win) / win
        envelope = np.convolve(envelope, kernel, mode="same")

    thresh = max(np.percentile(envelope, 90) * 0.25, 0.005)
    regions = _contiguous_regions(envelope > thresh)

    tones = []
    for s, e in regions:
        dur = (e - s) / sr
        if min_dur <= dur <= max_dur and _is_tonal(audio[s:e], sr, target_freq):
            tones.append((s, e))

    return _merge_close_regions(tones, min_gap=int(0.1 * sr))


def _is_tonal(
    seg: np.ndarray, sr: int, expected_freq: float, tol: float = 400.0,
) -> bool:
    """Check if *seg* is a pure tone near *expected_freq*."""
    if len(seg) < 256:
        return False
    n = len(seg)
    fft = np.abs(np.fft.rfft(seg * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    peak_freq = freqs[np.argmax(fft)]
    if abs(peak_freq - expected_freq) > tol:
        return False
    total = np.sum(fft ** 2)
    if total < 1e-10:
        return False
    mask = (freqs >= peak_freq - 100) & (freqs <= peak_freq + 100)
    return (np.sum(fft[mask] ** 2) / total) > 0.1


# ======================================================================
#  SIGNAL 3: FULL-FILE ASR TEXT MATCHING
# ======================================================================

def _match_stimuli_in_transcript(
    word_timeline: List[Dict[str, Any]],
    stimuli: List[str],
    similarity_threshold: float = 0.55,
) -> List[Dict[str, Any]]:
    """Fuzzy-match stimulus sentences against a word-timestamped transcript.

    Processes stimuli *in order*, advancing through the timeline so that
    matches never overlap and earlier stimuli cannot steal later positions.

    Returns
    -------
    list of dict
        Each dict: ``{"sentence_idx", "start_s", "end_s",
        "start_word_idx", "end_word_idx", "similarity"}``.
        Sorted by ``start_s``.
    """
    if not word_timeline:
        return []

    words_lower = [w.get("word", "").strip().lower() for w in word_timeline]
    n_total = len(words_lower)
    matches: List[Dict[str, Any]] = []
    search_from = 0

    for sent_idx, stimulus in enumerate(stimuli):
        stim_words = stimulus.lower().split()
        stim_set = set(stim_words)
        n_stim = len(stim_words)
        stim_text = " ".join(stim_words)
        if n_stim == 0:
            continue

        best_score = 0.0
        best_pos: Optional[Tuple[int, int]] = None

        # Sliding window sizes: n_stim-2 .. n_stim+3
        min_win = max(n_stim - 2, 1)
        max_win = n_stim + 4
        found_excellent = False

        for win in range(min_win, max_win + 1):
            if found_excellent:
                break
            for i in range(search_from, n_total - win + 1):
                # Quick pre-filter: at least 2 words in common
                window_words = words_lower[i : i + win]
                if len(stim_set & set(window_words)) < min(2, n_stim):
                    continue

                candidate = " ".join(window_words)
                m = SequenceMatcher(None, stim_text, candidate)
                # Fast upper-bound check
                if m.quick_ratio() < similarity_threshold:
                    continue
                score = m.ratio()
                if score > best_score:
                    best_score = score
                    best_pos = (i, i + win - 1)

                # Early termination on excellent match
                if best_score >= 0.85:
                    found_excellent = True
                    break

        if best_pos is not None and best_score >= similarity_threshold:
            si, ei = best_pos
            matches.append({
                "sentence_idx": sent_idx,
                "start_s": word_timeline[si]["start"],
                "end_s": word_timeline[ei]["end"],
                "start_word_idx": si,
                "end_word_idx": ei,
                "similarity": best_score,
            })
            # Advance past match + approximate response words
            search_from = ei + max(n_stim // 2, 3)
            logger.debug(
                "  stim %2d matched (%.2f) at [%.1f-%.1fs]",
                sent_idx + 1, best_score,
                word_timeline[si]["start"], word_timeline[ei]["end"],
            )
        else:
            logger.debug(
                "  stim %2d NOT matched (best %.2f)", sent_idx + 1, best_score,
            )

    matches.sort(key=lambda m: m["start_s"])
    return matches


# ======================================================================
#  RESPONSE EXTRACTION — TEXT MATCHING PATH
# ======================================================================

def _extract_responses_text_matching(
    audio: np.ndarray,
    sr: int,
    vad_regions: List[Tuple[int, int]],
    tones: List[Tuple[int, int]],
    stim_matches: List[Dict[str, Any]],
    word_timeline: List[Dict[str, Any]],
    config: SegmentationConfig,
) -> List[AudioSegment]:
    """Extract responses using stimulus text matches + tone anchors + VAD."""

    padding_before = int(config.padding_before_ms * sr / 1000)
    padding_after = int(config.padding_after_ms * sr / 1000)
    max_response_samples = int(config.max_response_duration_ms * sr / 1000)
    total_dur = len(audio) / sr

    # Fill in any missing stimulus positions
    stim_windows = _fill_stimulus_gaps(
        stim_matches, tones, sr, total_dur, config.expected_segments,
    )

    responses: List[AudioSegment] = []

    for i, win in enumerate(stim_windows):
        stim_end_s = win["end_s"]

        # Next stimulus start (or end of audio)
        if i + 1 < len(stim_windows):
            next_start_s = stim_windows[i + 1]["start_s"]
        else:
            next_start_s = total_dur

        # Find tone after this stimulus
        tone_end_s = _find_nearest_tone_after(
            tones, sr, stim_end_s, next_start_s,
        )

        # Response boundaries
        if tone_end_s is not None:
            resp_start_s = tone_end_s
        else:
            resp_start_s = stim_end_s + 0.5

        resp_end_s = next_start_s - 0.3  # small buffer before next stimulus

        # Clip to actual speech via VAD
        resp_start_samp, resp_end_samp, has_speech = _clip_to_vad(
            vad_regions, resp_start_s, resp_end_s, sr,
        )

        notes = ""
        if not has_speech:
            resp_start_samp = int(resp_start_s * sr)
            resp_end_samp = min(int((resp_start_s + 0.5) * sr), len(audio))
            notes = "no_response"

        # Apply padding
        start = max(0, resp_start_samp - padding_before)
        end = min(len(audio), resp_end_samp + padding_after)
        if end - start > max_response_samples:
            end = start + max_response_samples

        seg_audio = audio[start:end].copy()
        rms = (
            float(np.sqrt(np.mean(seg_audio ** 2))) if len(seg_audio) > 0 else 0.0
        )
        if rms < 0.003 and notes != "no_response":
            notes = "no_response"

        # Pre-transcription from word timeline
        pre_text = ""
        if word_timeline and has_speech and notes != "no_response":
            pre_text = _extract_text_in_window(
                word_timeline, resp_start_s, resp_end_s,
            )

        responses.append(AudioSegment(
            start_sample=start,
            end_sample=end,
            start_s=start / sr,
            end_s=end / sr,
            segment_type="response",
            sentence_number=0,  # assigned after sorting
            audio=seg_audio,
            energy_rms=rms,
            notes=notes,
            pre_transcription=pre_text,
        ))

    return responses


# ---------------------------------------------------------------------------
# Text-matching helpers
# ---------------------------------------------------------------------------

def _fill_stimulus_gaps(
    stim_matches: List[Dict[str, Any]],
    tones: List[Tuple[int, int]],
    sr: int,
    total_dur: float,
    expected: int,
) -> List[Dict[str, Any]]:
    """Produce a full list of *expected* stimulus windows.

    Matched stimuli are used directly; missing ones are estimated from
    neighbouring matches and nearby unassigned tones.
    """
    matched = {m["sentence_idx"]: m for m in stim_matches}
    sorted_tones = sorted(tones, key=lambda t: t[0])

    # Mark tones already associated with a matched stimulus
    used_tones: set = set()
    for m in stim_matches:
        for j, (ts, _te) in enumerate(sorted_tones):
            if m["end_s"] - 1.0 <= ts / sr <= m["end_s"] + 3.0:
                used_tones.add(j)
                break

    available = [
        (j, sorted_tones[j])
        for j in range(len(sorted_tones))
        if j not in used_tones
    ]
    avail_ptr = 0

    all_windows: List[Dict[str, Any]] = []
    for idx in range(expected):
        if idx in matched:
            all_windows.append(matched[idx])
            continue

        # Estimate from neighbours
        prev_end = 0.0
        next_start = total_dur
        for j in range(idx - 1, -1, -1):
            if j in matched:
                prev_end = matched[j]["end_s"] + 10.0
                break
        for j in range(idx + 1, expected):
            if j in matched:
                next_start = matched[j]["start_s"]
                break

        # Try unassigned tone in this gap
        tone_found = None
        for ai in range(avail_ptr, len(available)):
            _tj, (ts, te) = available[ai]
            t_s = ts / sr
            if prev_end - 5.0 <= t_s <= next_start:
                tone_found = (ts, te)
                avail_ptr = ai + 1
                break

        if tone_found is not None:
            est_end = tone_found[0] / sr
            est_start = max(0.0, est_end - 5.0)
        else:
            est_start = (prev_end + next_start) / 2 - 5.0
            est_end = (prev_end + next_start) / 2

        all_windows.append({
            "sentence_idx": idx,
            "start_s": est_start,
            "end_s": est_end,
            "interpolated": True,
        })

    return all_windows


def _find_nearest_tone_after(
    tones: List[Tuple[int, int]],
    sr: int,
    after_s: float,
    before_s: float,
) -> Optional[float]:
    """Return end-time (seconds) of first tone between *after_s* and *before_s*."""
    for ts, te in tones:
        t_start_s = ts / sr
        t_end_s = te / sr
        if after_s - 0.5 <= t_start_s <= before_s:
            return t_end_s
    return None


def _clip_to_vad(
    vad_regions: List[Tuple[int, int]],
    start_s: float,
    end_s: float,
    sr: int,
) -> Tuple[int, int, bool]:
    """Clip a time window to actual speech (VAD) within it.

    Returns ``(start_sample, end_sample, has_speech)``.
    """
    start_samp = int(start_s * sr)
    end_samp = int(end_s * sr)

    speech: List[Tuple[int, int]] = []
    for vs, ve in vad_regions:
        if ve > start_samp and vs < end_samp:
            cs = max(vs, start_samp)
            ce = min(ve, end_samp)
            if ce > cs:
                speech.append((cs, ce))

    if speech:
        return speech[0][0], speech[-1][1], True
    return start_samp, end_samp, False


def _extract_text_in_window(
    word_timeline: List[Dict[str, Any]],
    start_s: float,
    end_s: float,
) -> str:
    """Gather transcript words whose midpoint falls in [start_s, end_s]."""
    words = []
    for w in word_timeline:
        mid = (w.get("start", 0.0) + w.get("end", 0.0)) / 2.0
        if start_s <= mid <= end_s:
            words.append(w.get("word", "").strip())
    return " ".join(words).strip()


def _attach_pre_transcriptions(
    segments: List[AudioSegment],
    word_timeline: List[Dict[str, Any]],
) -> None:
    """Attach pre_transcription to each segment from the full-file ASR.

    For each response segment, extracts the words from *word_timeline*
    that fall within its time boundaries. This avoids per-segment
    re-transcription and provides better context.
    """
    for seg in segments:
        if seg.notes == "no_response":
            continue
        # Use a small buffer (0.3s) to capture words at segment edges
        text = _extract_text_in_window(
            word_timeline, seg.start_s - 0.3, seg.end_s + 0.3,
        )
        if text:
            seg.pre_transcription = text


# ======================================================================
#  RESPONSE EXTRACTION — FALLBACK (VAD + TONE GROUPING)
# ======================================================================

@dataclass
class _EITItem:
    """Internal: one stimulus-tone-response cycle."""
    regions: List[Tuple[int, int]]
    tone: Optional[Tuple[int, int]]
    item_start: int
    item_end: int


def _extract_responses_vad_tones(
    audio: np.ndarray,
    sr: int,
    vad_regions: List[Tuple[int, int]],
    tones: List[Tuple[int, int]],
    config: SegmentationConfig,
) -> List[AudioSegment]:
    """Fallback: group VAD regions by silence gaps, extract responses."""
    if not vad_regions:
        logger.warning("No VAD regions — returning empty")
        return []

    items = _group_into_items(vad_regions, sr, config, tones)
    logger.info("  Grouped into %d items", len(items))
    return _extract_responses_from_items(audio, sr, items, tones, config)


def _group_into_items(
    speech_regions: List[Tuple[int, int]],
    sr: int,
    config: SegmentationConfig,
    tones: List[Tuple[int, int]],
) -> List[_EITItem]:
    """Group speech regions into EIT items via adaptive gap threshold."""
    if len(speech_regions) < 2:
        return [_EITItem(
            regions=speech_regions,
            tone=None,
            item_start=speech_regions[0][0] if speech_regions else 0,
            item_end=speech_regions[0][1] if speech_regions else 0,
        )]

    gaps = [
        (speech_regions[i + 1][0] - speech_regions[i][1]) / sr
        for i in range(len(speech_regions) - 1)
    ]

    sorted_gaps = sorted(gaps, reverse=True)
    target_splits = config.expected_segments - 1

    if len(sorted_gaps) > target_splits:
        gap_thresh = max(
            (sorted_gaps[target_splits - 1] + sorted_gaps[target_splits]) / 2.0,
            2.0,
        )
    else:
        gap_thresh = 2.0

    logger.debug(
        "Gap threshold: %.2fs (range %.2f-%.2fs)",
        gap_thresh, min(gaps), max(gaps),
    )

    groups: List[List[Tuple[int, int]]] = [[speech_regions[0]]]
    for i, gap in enumerate(gaps):
        if gap >= gap_thresh:
            groups.append([speech_regions[i + 1]])
        else:
            groups[-1].append(speech_regions[i + 1])

    items: List[_EITItem] = []
    for group in groups:
        item_start = group[0][0]
        item_end = group[-1][1]
        margin = int(0.5 * sr)
        item_tone = None
        for ts, te in tones:
            if ts >= item_start - margin and te <= item_end + margin:
                item_tone = (ts, te)
                break
        items.append(_EITItem(
            regions=group, tone=item_tone,
            item_start=item_start, item_end=item_end,
        ))

    return items


def _extract_responses_from_items(
    audio: np.ndarray,
    sr: int,
    items: List[_EITItem],
    tones: List[Tuple[int, int]],
    config: SegmentationConfig,
) -> List[AudioSegment]:
    """Extract the response portion from each EIT item."""
    padding_before = int(config.padding_before_ms * sr / 1000)
    padding_after = int(config.padding_after_ms * sr / 1000)
    max_resp = int(config.max_response_duration_ms * sr / 1000)

    responses: List[AudioSegment] = []

    for item in items:
        resp_start = resp_end = None
        notes = ""

        if item.tone is not None:
            tone_end = item.tone[1]
            # Regions that start after the tone
            post_tone = [
                (s, e) for s, e in item.regions
                if s >= tone_end - int(0.05 * sr)
            ]
            if post_tone:
                resp_start, resp_end = post_tone[0][0], post_tone[-1][1]
            else:
                # No separate region after tone; check if a region *spans*
                # the tone (stimulus+response merged into one VAD region).
                spanning = [
                    (s, e) for s, e in item.regions
                    if s < tone_end and e > tone_end + int(0.1 * sr)
                ]
                if spanning:
                    resp_start = tone_end
                    resp_end = spanning[-1][1]
                else:
                    resp_start = tone_end
                    resp_end = min(tone_end + int(0.5 * sr), len(audio))
                    notes = "no_response"

        elif len(item.regions) >= 2:
            resp_start, resp_end = item.regions[-1]

        else:
            region_start, region_end = item.regions[0]
            dur = (region_end - region_start) / sr
            if dur < 2.0:
                resp_start, resp_end = region_start, region_end
            else:
                sp = _find_tone_boundary_in_region(
                    audio, sr, region_start, region_end, config,
                )
                if sp is not None:
                    resp_start, resp_end = sp, region_end
                else:
                    mid = (region_start + region_end) // 2
                    resp_start, resp_end = mid, region_end

        if resp_start is None:
            continue

        start = max(0, resp_start - padding_before)
        end = min(len(audio), resp_end + padding_after)
        if end - start > max_resp:
            end = start + max_resp

        seg_audio = audio[start:end].copy()
        rms = (
            float(np.sqrt(np.mean(seg_audio ** 2))) if len(seg_audio) > 0 else 0.0
        )
        if rms < 0.005 and notes != "no_response":
            notes = "no_response"

        responses.append(AudioSegment(
            start_sample=start,
            end_sample=end,
            start_s=start / sr,
            end_s=end / sr,
            segment_type="response",
            sentence_number=0,
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
    """Look for a tone-like boundary inside a single speech region."""
    from scipy import signal as scipy_signal

    seg = audio[region_start:region_end]
    if len(seg) < int(0.1 * sr):
        return None

    target = config.tone_freq_hz
    low = max(target - 300, 100)
    high = min(target + 300, sr / 2 - 1)
    nyq = sr / 2.0
    b, a = scipy_signal.butter(4, [low / nyq, high / nyq], btype="band")
    filtered = scipy_signal.filtfilt(b, a, seg)
    envelope = np.abs(scipy_signal.hilbert(filtered))
    win = int(0.02 * sr)
    if win > 0:
        kernel = np.ones(win) / win
        envelope = np.convolve(envelope, kernel, mode="same")

    thresh = np.percentile(envelope, 80) * 0.5
    if thresh < 0.003:
        return None

    for s, e in _contiguous_regions(envelope > thresh):
        dur = (e - s) / sr
        if config.tone_min_duration_ms / 2000.0 <= dur <= config.tone_max_duration_ms / 1000.0:
            return region_start + e

    return None


# ======================================================================
#  HELPERS
# ======================================================================

def _contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous True regions -> ``[(start, end), ...]``."""
    if len(mask) == 0:
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [len(mask)]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _merge_close_regions(
    regions: List[Tuple[int, int]], min_gap: int,
) -> List[Tuple[int, int]]:
    """Merge regions separated by less than *min_gap* samples."""
    if not regions:
        return []
    merged = [regions[0]]
    for s, e in regions[1:]:
        ps, pe = merged[-1]
        if s - pe <= min_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged