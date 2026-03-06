"""Speaker diarization module — Phase 2 (Advanced).

Uses pyannote.audio speaker diarization + speaker embeddings to separate:
  - **Stimulus speaker** (native Spanish, played through speakers) 
  - **Response speaker** (non-native, recorded directly into mic)

This enables more precise response boundaries beyond VAD + tone detection,
since we can directly select only audio turns attributed to the non-native
participant.

Pipeline:
  1. Run pyannote SpeakerDiarization on the preprocessed WAV
  2. Collect all speaker turns → (start, end, speaker_label)
  3. Identify the "response speaker" using tone-anchor voting:
       - For each tone position, the speaker active just AFTER the tone
         belongs to the response track
       - The label with the most post-tone votes = response speaker
  4. Merge consecutive short turns (removes fragmentation artefacts)  
  5. Group merged response turns into 30 EIT items using tone/VAD anchors
  6. Return List[AudioSegment], structured identically to Phase 1 output

Requirements:
    pip install pyannote.audio torch

Hugging Face token:
    The pyannote pipeline is gated — you must:
      1. Accept terms at https://hf.co/pyannote/speaker-diarization-3.1
      2. Accept terms at https://hf.co/pyannote/segmentation-3.0
      3. Pass your HF token via DiarizationConfig.hf_token or the 
         environment variable HUGGINGFACE_TOKEN.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wav_io
import tempfile

from .config import SegmentationConfig, TARGET_SENTENCES
from .segmentation import AudioSegment, detect_tones, _silero_vad_detect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class DiarizationSegment:
    """A raw diarization turn from pyannote."""
    start_s: float
    end_s: float
    speaker_label: str


@dataclass
class DiarizationResult:
    """Full diarization output for one audio file."""
    turns: List[DiarizationSegment]
    response_speaker: str                  # pyannote label for non-native speaker
    stimulus_speaker: Optional[str]        # pyannote label for native stimulus
    num_speakers_detected: int
    method: str = "pyannote"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core diarization
# ---------------------------------------------------------------------------

def run_diarization(
    audio: np.ndarray,
    sr: int,
    hf_token: str,
    num_speakers: int = 2,
    min_speakers: int = 2,
    max_speakers: int = 2,
) -> DiarizationResult:
    """Run pyannote speaker diarization on an audio array.

    Parameters
    ----------
    audio : np.ndarray
        Preprocessed float32 mono audio.
    sr : int
        Sample rate (16 000 recommended).
    hf_token : str
        Hugging Face access token (model is gated).
    num_speakers : int
        Fixed number of speakers to detect. Set to 0 for automatic.
    min_speakers / max_speakers : int
        Bounds used when num_speakers=0.  Defaults to 2/2 since EIT
        always has exactly two speakers (stimulus + participant).

    Returns
    -------
    DiarizationResult
    """
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio is required for speaker diarization.\n"
            "Install it with:  pip install pyannote.audio"
        ) from exc

    # Clear GPU/MPS cache from any previous run before loading the pipeline.
    # This is the primary fix for MPS OOM when processing multiple files
    # sequentially — the previous run's allocations are released first.
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Loading pyannote SpeakerDiarization-3.1 pipeline …")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    # Choose device — prefer MPS on Apple Silicon, fall back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("  Using Apple MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.info("  Using CPU (no GPU available)")

    pipeline = pipeline.to(device)

    # Pyannote expects a file path or dict with waveform tensor + sample_rate
    try:
        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, T)
        audio_input = {"waveform": waveform, "sample_rate": sr}
    except Exception:
        # Fallback: write temp wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        _write_wav(tmp_path, audio, sr)
        audio_input = tmp_path

    def _run_pipeline(pipe, inp):
        if num_speakers > 0:
            return pipe(inp, num_speakers=num_speakers,
                        min_speakers=min_speakers, max_speakers=max_speakers)
        return pipe(inp, min_speakers=min_speakers, max_speakers=max_speakers)

    logger.info("Running pyannote diarization (num_speakers=%s)…", num_speakers or "auto")
    try:
        diarization = _run_pipeline(pipeline, audio_input)
    except RuntimeError as oom_err:
        if "out of memory" in str(oom_err) and device.type != "cpu":
            logger.warning(
                "  MPS/GPU out of memory — retrying on CPU. "
                "To avoid this, close other GPU-heavy apps first."
            )
            # Free GPU memory before switching
            pipeline = pipeline.to(torch.device("cpu"))
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            device = torch.device("cpu")
            diarization = _run_pipeline(pipeline, audio_input)
        else:
            raise

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(DiarizationSegment(
            start_s=turn.start,
            end_s=turn.end,
            speaker_label=speaker,
        ))

    # Clean up temp file if we created one
    if isinstance(audio_input, str):
        try:
            os.unlink(audio_input)
        except OSError:
            pass

    # Explicitly free GPU/MPS memory after inference so the next
    # participant's run starts with a clean slate.
    try:
        pipeline = pipeline.to(torch.device("cpu"))
        del pipeline
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    speaker_labels = sorted({t.speaker_label for t in turns})
    num_detected = len(speaker_labels)
    logger.info("  %d speakers detected: %s", num_detected, speaker_labels)

    return DiarizationResult(
        turns=turns,
        response_speaker="",          # filled in by identify_response_speaker
        stimulus_speaker=None,
        num_speakers_detected=num_detected,
    )


# ---------------------------------------------------------------------------
# Identify the response (non-native) speaker
# ---------------------------------------------------------------------------

def identify_response_speaker(
    result: DiarizationResult,
    audio: np.ndarray,
    sr: int,
    config: SegmentationConfig,
    window_after_tone_s: float = 2.0,
) -> DiarizationResult:
    """Determine which pyannote speaker label corresponds to the non-native
    participant responding to each stimulus.

    Strategy (tone-free — pure diarization signals)
    ------------------------------------------------
    1. **Per-turn average RMS** — The participant speaks directly into the
       mic, so their turns have consistently *higher* RMS than the stimulus
       audio which leaks from room speakers at lower relative level.
    2. **Alternation-pattern voting** — In EIT the speakers strictly
       alternate (stimulus → response → stimulus → …).  The speaker who
       most often *follows* the other is the response speaker.
    3. Results from both signals are combined; if they agree the confidence
       is high.  On disagreement, per-turn RMS wins (stronger physical
       signal).

    Returns
    -------
    DiarizationResult with ``response_speaker`` and ``stimulus_speaker`` set.
    """
    speaker_labels = sorted({t.speaker_label for t in result.turns})

    if not speaker_labels:
        logger.warning("No speaker turns found in diarization result.")
        result.response_speaker = "SPEAKER_00"
        return result

    if len(speaker_labels) == 1:
        result.response_speaker = speaker_labels[0]
        return result

    # ------------------------------------------------------------------
    # Signal 1: Per-turn mean RMS  (direct mic → higher RMS)
    # ------------------------------------------------------------------
    per_turn_rms: Dict[str, List[float]] = {lbl: [] for lbl in speaker_labels}
    for turn in result.turns:
        t_start = int(turn.start_s * sr)
        t_end = min(int(turn.end_s * sr), len(audio))
        chunk = audio[t_start:t_end]
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
        per_turn_rms[turn.speaker_label].append(rms)

    mean_rms = {lbl: float(np.mean(vals)) if vals else 0.0
                for lbl, vals in per_turn_rms.items()}
    rms_winner = max(mean_rms, key=lambda k: mean_rms[k])
    logger.info("  Per-turn mean RMS: %s → winner: %s", mean_rms, rms_winner)

    # ------------------------------------------------------------------
    # Signal 2: Alternation pattern  (response follows stimulus)
    # ------------------------------------------------------------------
    sorted_turns = sorted(result.turns, key=lambda t: t.start_s)
    follower_count: Dict[str, int] = {lbl: 0 for lbl in speaker_labels}
    for i in range(len(sorted_turns) - 1):
        cur_lbl = sorted_turns[i].speaker_label
        nxt_lbl = sorted_turns[i + 1].speaker_label
        if cur_lbl != nxt_lbl:
            follower_count[nxt_lbl] += 1

    alt_winner = max(follower_count, key=lambda k: follower_count[k])
    logger.info("  Alternation follower counts: %s → winner: %s",
                follower_count, alt_winner)

    # ------------------------------------------------------------------
    # Combine: agree → high confidence;  disagree → trust RMS
    # ------------------------------------------------------------------
    if rms_winner == alt_winner:
        response_speaker = rms_winner
        logger.info("  Both signals agree → response speaker: %s (high confidence)",
                    response_speaker)
    else:
        response_speaker = rms_winner
        logger.info("  Signals disagree (RMS=%s, alt=%s) → trusting RMS: %s",
                    rms_winner, alt_winner, response_speaker)

    other_speakers = [lbl for lbl in speaker_labels if lbl != response_speaker]
    result.response_speaker = response_speaker
    result.stimulus_speaker = other_speakers[0] if other_speakers else None

    logger.info(
        "  Response speaker: %s | Stimulus/other: %s",
        result.response_speaker, result.stimulus_speaker,
    )
    return result


# ---------------------------------------------------------------------------
# Build AudioSegments from diarization turns
# ---------------------------------------------------------------------------

def segments_from_diarization(
    result: DiarizationResult,
    audio: np.ndarray,
    sr: int,
    config: SegmentationConfig,
    min_duration_s: float = 0.3,
    merge_gap_s: float = 0.8,
    n_items: int = 30,
) -> List[AudioSegment]:
    """Convert diarization turns into List[AudioSegment] numbered 1–N.

    Pure-diarization approach (no tone dependency):
    1. Filter to response-speaker turns only.
    2. Merge adjacent turns separated by < ``merge_gap_s`` seconds.
    3. Remove very short turns (< ``min_duration_s``).
    4. If more than ``n_items`` merged turns remain, keep the N longest.
    5. Number chronologically 1–N.
    """
    # Step 1: filter to response speaker
    response_turns = [
        t for t in result.turns
        if t.speaker_label == result.response_speaker
    ]
    logger.info(
        "  Diarization → %d raw turns for response speaker '%s'",
        len(response_turns), result.response_speaker,
    )
    if not response_turns:
        logger.warning("No response turns found — returning empty segment list.")
        return []

    # Step 2: merge close turns
    response_turns.sort(key=lambda t: t.start_s)
    merged: List[DiarizationSegment] = []
    cur = response_turns[0]
    for nxt in response_turns[1:]:
        if nxt.start_s - cur.end_s <= merge_gap_s:
            cur = DiarizationSegment(cur.start_s, nxt.end_s, cur.speaker_label)
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)

    # Step 3: remove very short segments
    merged = [t for t in merged if (t.end_s - t.start_s) >= min_duration_s]
    logger.info("  After merge + filter: %d candidate response turns", len(merged))

    # Step 4: if too many, keep the N longest (sorted back by time)
    if len(merged) > n_items:
        logger.info(
            "  %d turns > %d items — keeping %d longest turns",
            len(merged), n_items, n_items,
        )
        merged = sorted(merged, key=lambda t: t.end_s - t.start_s, reverse=True)[:n_items]
        merged.sort(key=lambda t: t.start_s)

    # Step 5: build AudioSegments, numbered chronologically
    segments: List[AudioSegment] = []
    for i, turn in enumerate(merged):
        start_samp = int(turn.start_s * sr)
        end_samp = min(int(turn.end_s * sr), len(audio))
        seg_audio = audio[start_samp:end_samp].copy()
        rms = float(np.sqrt(np.mean(seg_audio ** 2))) if len(seg_audio) > 0 else 0.0
        segments.append(AudioSegment(
            start_sample=start_samp,
            end_sample=end_samp,
            start_s=turn.start_s,
            end_s=turn.end_s,
            segment_type="response",
            audio=seg_audio,
            energy_rms=rms,
            sentence_number=i + 1,
        ))

    logger.info("  Final: %d response segments", len(segments))
    return segments


# ---------------------------------------------------------------------------
# Group turns into exactly N EIT items (tone-guided)
# ---------------------------------------------------------------------------

def _group_into_eit_items(
    turns: List[DiarizationSegment],
    tones: List[Tuple[int, int]],
    audio: np.ndarray,
    sr: int,
    config: SegmentationConfig,
    n_items: int = 30,
) -> List[AudioSegment]:
    """Assign each response turn to one of N EIT items using tone boundaries.

    If tones provide clear item boundaries (>= n_items//2 tones detected),
    use them as dividers.  Otherwise fall back to uniform time-bucketing.
    """
    tone_times_s = [(s / sr, e / sr) for s, e in tones]

    def _make_seg(turn: DiarizationSegment) -> AudioSegment:
        start_samp = int(turn.start_s * sr)
        end_samp = min(int(turn.end_s * sr), len(audio))
        seg_audio = audio[start_samp:end_samp].copy()
        rms = float(np.sqrt(np.mean(seg_audio ** 2))) if len(seg_audio) > 0 else 0.0
        return AudioSegment(
            start_sample=start_samp,
            end_sample=end_samp,
            start_s=turn.start_s,
            end_s=turn.end_s,
            segment_type="response",
            audio=seg_audio,
            energy_rms=rms,
        )

    if len(tone_times_s) >= n_items // 2:
        # Use tone starts as item boundaries — each response is the first
        # response turn after a tone
        tone_starts = sorted(t[0] for t in tone_times_s)

        # Build item→[turns] mapping.
        # NOTE: do NOT use `break` inside the inner loop — it stops scanning
        # early and leaves turns unassigned for tones after the break point.
        item_turns: List[List[DiarizationSegment]] = [[] for _ in range(len(tone_starts))]

        for turn in turns:
            best_tone_idx = None
            for i, ts in enumerate(tone_starts):
                # Turn starts at or after this tone → candidate assignment
                if turn.start_s >= ts - 0.5:
                    best_tone_idx = i
                # Do NOT break here — keep scanning for a later (better) tone
            if best_tone_idx is not None:
                item_turns[best_tone_idx].append(turn)

        # Flatten: one representative segment per item (longest turn)
        segments: List[AudioSegment] = []
        for item_group in item_turns:
            if not item_group:
                continue
            best = max(item_group, key=lambda t: t.end_s - t.start_s)
            segments.append(_make_seg(best))

        if len(segments) >= n_items // 2:
            return segments

    # Fallback: take the N longest turns
    logger.info(
        "  Falling back to N-longest strategy (insufficient tone anchors)."
    )
    turns_sorted_by_length = sorted(turns, key=lambda t: t.end_s - t.start_s, reverse=True)
    top_n = sorted(turns_sorted_by_length[:n_items], key=lambda t: t.start_s)
    return [_make_seg(t) for t in top_n]


# ---------------------------------------------------------------------------
# Full diarization-based segmentation entry point
# ---------------------------------------------------------------------------

def segment_with_diarization(
    audio: np.ndarray,
    sr: int,
    config: SegmentationConfig,
    hf_token: str,
    num_speakers: int = 2,
    min_speakers: int = 2,
    max_speakers: int = 2,
    merge_gap_s: float = 0.8,
    word_timeline: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[AudioSegment], DiarizationResult]:
    """End-to-end: diarize → identify response speaker → extract segments.

    Parameters
    ----------
    audio, sr : preprocessed audio
    config : SegmentationConfig (reused for tone detection parameters)
    hf_token : Hugging Face access token
    num_speakers : fixed number of speakers (0 = auto)
    min_speakers / max_speakers : bounds (EIT = always 2)
    merge_gap_s : merge response turns closer than this (0.8 s for non-native)
    word_timeline : optional full-file ASR words for pre-transcription enrichment

    Returns
    -------
    (segments, diarization_result)
    """
    # 1. Diarize
    diar_result = run_diarization(
        audio, sr, hf_token,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # 2. Identify response speaker
    diar_result = identify_response_speaker(diar_result, audio, sr, config)

    # 3. Build segments
    segments = segments_from_diarization(
        diar_result, audio, sr, config, merge_gap_s=merge_gap_s,
    )

    # 4. Attach pre-transcriptions
    if word_timeline:
        from .segmentation import _attach_pre_transcriptions
        _attach_pre_transcriptions(segments, word_timeline)
        n_enriched = sum(1 for s in segments if s.pre_transcription)
        logger.info("  Pre-transcriptions attached: %d / %d", n_enriched, len(segments))

    logger.info(
        "Diarization segmentation complete: %d segments (response speaker: %s)",
        len(segments), diar_result.response_speaker,
    )
    return segments, diar_result


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare_segmentation_methods(
    phase1_segments: List[AudioSegment],
    diar_segments: List[AudioSegment],
    audio: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    """Compute comparison metrics between Phase 1 and diarization segments.

    Returns a dict with keys:
      phase1_count, diar_count,
      phase1_total_s, diar_total_s,
      phase1_mean_dur_s, diar_mean_dur_s,
      phase1_no_response, diar_no_response,
      overlap_ratio_mean,  # how much each pair overlaps (chronological order)
      boundary_delta_mean_s,  # mean |start difference| between pairs
    """
    n1 = len(phase1_segments)
    n2 = len(diar_segments)
    paired = min(n1, n2)

    def durations(segs):
        return [s.end_s - s.start_s for s in segs]

    p1_dur = durations(phase1_segments)
    d_dur = durations(diar_segments)

    overlap_ratios = []
    boundary_deltas = []
    for i in range(paired):
        s1, s2 = phase1_segments[i], diar_segments[i]
        ov = _overlap_s(s1.start_s, s1.end_s, s2.start_s, s2.end_s)
        union = max(s1.end_s, s2.end_s) - min(s1.start_s, s2.start_s)
        overlap_ratios.append(ov / union if union > 0 else 0.0)
        boundary_deltas.append(abs(s1.start_s - s2.start_s))

    return {
        "phase1_count": n1,
        "diar_count": n2,
        "phase1_total_s": sum(p1_dur),
        "diar_total_s": sum(d_dur),
        "phase1_mean_dur_s": float(np.mean(p1_dur)) if p1_dur else 0.0,
        "diar_mean_dur_s": float(np.mean(d_dur)) if d_dur else 0.0,
        "phase1_no_response": sum(1 for s in phase1_segments if s.notes == "no_response"),
        "diar_no_response": sum(1 for s in diar_segments if s.notes == "no_response"),
        "paired_segments": paired,
        "overlap_ratio_mean": float(np.mean(overlap_ratios)) if overlap_ratios else 0.0,
        "boundary_delta_mean_s": float(np.mean(boundary_deltas)) if boundary_deltas else 0.0,
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _overlap_s(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write float32 audio to a 16-bit PCM WAV file."""
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    wav_io.write(path, sr, pcm)