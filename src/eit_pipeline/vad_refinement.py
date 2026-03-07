"""Stage 5 — VAD Boundary Refinement.

Refines response segment boundaries using frame-level VAD analysis.
Expands the window slightly, then trims to the first/last voiced frames.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

from .audio_io import AudioData
from .config import VADRefinementConfig
from .response_window import ResponseSegment

logger = logging.getLogger("eit_pipeline.vad_refinement")


def refine_boundaries(
    responses: List[ResponseSegment],
    recording: AudioData,
    config: Optional[VADRefinementConfig] = None,
) -> List[ResponseSegment]:
    """Refine response segment boundaries using frame-level VAD.

    Procedure for each response:
    1. Expand response window by ±expand_window_s
    2. Run frame-level VAD on expanded window
    3. Trim to first voiced frame and last voiced frame

    Parameters
    ----------
    responses : list of ResponseSegment
        Response windows from Stage 4.
    recording : AudioData
        Full recording audio data.
    config : VADRefinementConfig, optional
        Refinement parameters.

    Returns
    -------
    list of ResponseSegment
        Refined response segments.
    """
    if config is None:
        config = VADRefinementConfig()

    # Load Silero VAD for frame-level analysis
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )

    sr = recording.sample_rate

    refined = []
    for resp in responses:
        if resp.response_end_s == 0.0:
            # No response detected — pass through
            refined.append(resp)
            continue

        # Expand window
        exp_start = max(0, resp.response_start_s - config.expand_window_s)
        exp_end = min(
            recording.duration_s,
            resp.response_end_s + config.expand_window_s,
        )

        # Extract expanded audio segment
        start_sample = int(exp_start * sr)
        end_sample = int(exp_end * sr)
        segment_audio = recording.waveform[start_sample:end_sample]

        if len(segment_audio) == 0:
            refined.append(resp)
            continue

        # Run frame-level VAD
        frame_probs = _frame_level_vad(
            segment_audio, sr, model, config.frame_duration_ms,
        )

        if len(frame_probs) == 0:
            refined.append(resp)
            continue

        # Find first and last voiced frames
        voiced_mask = frame_probs >= config.vad_threshold
        voiced_indices = np.where(voiced_mask)[0]

        if len(voiced_indices) == 0:
            # No voiced frames found — mark as no response
            logger.debug(
                "No voiced frames for stimulus %d — clearing response",
                resp.sentence_id,
            )
            resp.response_start_s = 0.0
            resp.response_end_s = 0.0
            resp.confidence = 0.0
            refined.append(resp)
            continue

        # Convert frame indices to time
        frame_duration_s = config.frame_duration_ms / 1000.0
        first_voiced_s = exp_start + voiced_indices[0] * frame_duration_s
        last_voiced_s = exp_start + (voiced_indices[-1] + 1) * frame_duration_s

        # Update boundaries
        resp.response_start_s = first_voiced_s
        resp.response_end_s = last_voiced_s

        logger.debug(
            "Refined stimulus %d: %.2f-%.2fs (from expanded %.2f-%.2fs)",
            resp.sentence_id, first_voiced_s, last_voiced_s,
            exp_start, exp_end,
        )
        refined.append(resp)

    logger.info("Refined %d response boundaries", len(refined))
    return refined


def _frame_level_vad(
    audio: np.ndarray,
    sr: int,
    model: torch.nn.Module,
    frame_duration_ms: int,
) -> np.ndarray:
    """Compute per-frame voice activity probabilities.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform (mono, float32).
    sr : int
        Sample rate.
    model : torch.nn.Module
        Loaded Silero VAD model.
    frame_duration_ms : int
        Duration of each analysis frame.

    Returns
    -------
    np.ndarray
        Array of VAD probabilities, one per frame.
    """
    frame_samples = int(sr * frame_duration_ms / 1000)
    n_frames = len(audio) // frame_samples

    if n_frames == 0:
        return np.array([])

    # Reset model state
    model.reset_states()

    probs = []
    audio_tensor = torch.from_numpy(audio).float()

    for i in range(n_frames):
        start = i * frame_samples
        end = start + frame_samples
        chunk = audio_tensor[start:end]

        # Silero VAD expects specific chunk sizes (512 for 16kHz)
        # Pad if needed
        if len(chunk) < 512:
            chunk = torch.nn.functional.pad(chunk, (0, 512 - len(chunk)))

        # Get probability
        with torch.no_grad():
            prob = model(chunk.unsqueeze(0), sr)
        probs.append(float(prob))

    return np.array(probs)
