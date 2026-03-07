"""Stage 3 — Silence / Speech Detection.

Detects speech segments in the recording using Silero VAD.
Returns a list of speech regions with start/end timestamps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .audio_io import AudioData
from .config import SilenceDetectionConfig

logger = logging.getLogger("eit_pipeline.silence_detection")


@dataclass
class SpeechSegment:
    """A detected speech segment."""
    start_s: float
    end_s: float
    confidence: float = 1.0


def detect_speech(
    recording: AudioData,
    config: Optional[SilenceDetectionConfig] = None,
) -> List[SpeechSegment]:
    """Detect speech regions in a recording using Silero VAD.

    Parameters
    ----------
    recording : AudioData
        Full recording audio data.
    config : SilenceDetectionConfig, optional
        Detection parameters.

    Returns
    -------
    list of SpeechSegment
        Detected speech segments sorted by start time.
    """
    if config is None:
        config = SilenceDetectionConfig()

    logger.info("Running speech detection (backend=%s)", config.vad_backend)

    segments = _silero_vad(recording, config)

    # Filter short segments
    segments = [
        s for s in segments
        if (s.end_s - s.start_s) >= config.min_segment_duration_s
    ]

    # Merge nearby segments
    segments = _merge_segments(segments, config.merge_gap_s)

    logger.info("Detected %d speech segments", len(segments))
    return segments


def _silero_vad(
    recording: AudioData,
    config: SilenceDetectionConfig,
) -> List[SpeechSegment]:
    """Run Silero VAD on the recording."""
    # Load Silero VAD model
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]

    # Silero VAD expects 16 kHz audio
    sr = recording.sample_rate
    if sr != 16_000:
        import librosa
        waveform = librosa.resample(recording.waveform, orig_sr=sr, target_sr=16_000)
        sr = 16_000
    else:
        waveform = recording.waveform

    # Convert to torch tensor
    audio_tensor = torch.from_numpy(waveform).float()

    # Run VAD
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=config.silero_threshold,
        min_speech_duration_ms=config.silero_min_speech_duration_ms,
        min_silence_duration_ms=config.silero_min_silence_duration_ms,
        window_size_samples=config.silero_window_size_samples,
        sampling_rate=sr,
        return_seconds=False,
    )

    segments = []
    for ts in speech_timestamps:
        start_s = ts["start"] / sr
        end_s = ts["end"] / sr
        segments.append(SpeechSegment(start_s=start_s, end_s=end_s))

    logger.info("Silero VAD raw segments: %d", len(segments))
    return segments


def _merge_segments(
    segments: List[SpeechSegment],
    max_gap_s: float,
) -> List[SpeechSegment]:
    """Merge speech segments that are closer than max_gap_s."""
    if not segments:
        return segments

    merged = [SpeechSegment(
        start_s=segments[0].start_s,
        end_s=segments[0].end_s,
        confidence=segments[0].confidence,
    )]

    for seg in segments[1:]:
        if seg.start_s - merged[-1].end_s <= max_gap_s:
            merged[-1].end_s = seg.end_s
            merged[-1].confidence = max(merged[-1].confidence, seg.confidence)
        else:
            merged.append(SpeechSegment(
                start_s=seg.start_s,
                end_s=seg.end_s,
                confidence=seg.confidence,
            ))

    return merged
