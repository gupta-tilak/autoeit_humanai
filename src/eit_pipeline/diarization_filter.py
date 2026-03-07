"""Stage 6 — Optional Diarization Filter.

If enabled, runs speaker diarization to identify which segments belong to
the participant speaker and filters out segments dominated by the stimulus
speaker.

Does NOT assume which speaker label corresponds to the participant.  Instead,
analyses which speaker dominates the detected response windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio_io import AudioData
from .config import DiarizationConfig
from .response_window import ResponseSegment

logger = logging.getLogger("eit_pipeline.diarization_filter")


@dataclass
class DiarizationResult:
    """Result from speaker diarization."""
    participant_label: str
    turns: List[Dict]  # [{"speaker": str, "start": float, "end": float}, ...]
    speaker_stats: Dict[str, float]  # speaker_label -> total_duration


def filter_by_diarization(
    responses: List[ResponseSegment],
    recording: AudioData,
    config: Optional[DiarizationConfig] = None,
) -> Tuple[List[ResponseSegment], Optional[DiarizationResult]]:
    """Filter response segments using speaker diarization.

    Parameters
    ----------
    responses : list of ResponseSegment
        Response windows from previous stages.
    recording : AudioData
        Full recording audio.
    config : DiarizationConfig, optional
        Diarization parameters.

    Returns
    -------
    tuple
        (filtered_responses, diarization_result)
        If diarization is disabled, returns (responses, None).
    """
    if config is None:
        config = DiarizationConfig()

    if not config.enabled:
        logger.info("Diarization disabled — passing through all responses")
        return responses, None

    # Check for HF token
    hf_token = config.hf_token
    if not hf_token:
        import os
        hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    if not hf_token:
        logger.warning(
            "No HuggingFace token found — diarization requires authentication. "
            "Set HUGGINGFACE_TOKEN or HF_TOKEN environment variable."
        )
        return responses, None

    logger.info("Running speaker diarization...")

    try:
        diarization = _run_pyannote_diarization(recording, config, hf_token)
    except Exception as e:
        logger.error("Diarization failed: %s", e)
        return responses, None

    # Identify participant speaker from response windows
    participant_label = _identify_participant(diarization.turns, responses)
    diarization.participant_label = participant_label
    logger.info("Identified participant speaker: %s", participant_label)

    # Filter responses: remove those dominated by stimulus speaker
    filtered = []
    for resp in responses:
        if resp.response_end_s == 0.0:
            filtered.append(resp)
            continue

        # Check speaker overlap with response window
        participant_overlap = _speaker_overlap(
            diarization.turns,
            participant_label,
            resp.response_start_s,
            resp.response_end_s,
        )

        total_overlap = sum(
            _speaker_overlap(diarization.turns, spk, resp.response_start_s, resp.response_end_s)
            for spk in diarization.speaker_stats
        )

        if total_overlap > 0 and participant_overlap / total_overlap < 0.3:
            logger.warning(
                "Response %d dominated by stimulus speaker — filtering out "
                "(participant overlap: %.1f%%)",
                resp.sentence_id,
                100 * participant_overlap / total_overlap,
            )
            resp.response_start_s = 0.0
            resp.response_end_s = 0.0
            resp.confidence = 0.0

        filtered.append(resp)

    logger.info("Diarization filter complete")
    return filtered, diarization


def _run_pyannote_diarization(
    recording: AudioData,
    config: DiarizationConfig,
    hf_token: str,
) -> DiarizationResult:
    """Run pyannote speaker diarization."""
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError:
        raise ImportError(
            "pyannote.audio is required for diarization. "
            "Install with: pip install pyannote.audio"
        )

    import tempfile
    from .audio_io import save_audio

    # Save to temp file for pyannote
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    save_audio(recording.waveform, recording.sample_rate, tmp_path)

    try:
        pipeline = PyannotePipeline.from_pretrained(
            config.model_name,
            use_auth_token=hf_token,
        )

        diarization_output = pipeline(
            tmp_path,
            num_speakers=config.num_speakers if config.num_speakers > 0 else None,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
        )
    finally:
        import os
        os.unlink(tmp_path)

    # Parse diarization output
    turns = []
    speaker_durations: Dict[str, float] = {}

    for turn, _, speaker in diarization_output.itertracks(yield_label=True):
        duration = turn.end - turn.start
        if duration < config.min_turn_duration_s:
            continue

        turns.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
        })
        speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

    return DiarizationResult(
        participant_label="",  # will be set later
        turns=turns,
        speaker_stats=speaker_durations,
    )


def _identify_participant(
    turns: List[Dict],
    responses: List[ResponseSegment],
) -> str:
    """Identify which speaker label corresponds to the participant.

    The participant is the speaker who dominates the response windows.
    """
    speaker_response_time: Dict[str, float] = {}

    for resp in responses:
        if resp.response_end_s == 0.0:
            continue

        for turn in turns:
            overlap = _time_overlap(
                resp.response_start_s, resp.response_end_s,
                turn["start"], turn["end"],
            )
            if overlap > 0:
                spk = turn["speaker"]
                speaker_response_time[spk] = speaker_response_time.get(spk, 0) + overlap

    if not speaker_response_time:
        # Fallback: use the speaker with less total time (stimulus speaker talks more)
        return ""

    return max(speaker_response_time, key=speaker_response_time.get)


def _speaker_overlap(
    turns: List[Dict],
    speaker: str,
    start_s: float,
    end_s: float,
) -> float:
    """Calculate total overlap between a speaker and a time window."""
    total = 0.0
    for turn in turns:
        if turn["speaker"] != speaker:
            continue
        overlap = _time_overlap(start_s, end_s, turn["start"], turn["end"])
        total += overlap
    return total


def _time_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Calculate overlap between two time intervals."""
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    return max(0.0, overlap_end - overlap_start)
