"""Stage 1 — Audio Loading and Preprocessing.

Loads a full participant recording, converts to mono, normalises amplitude,
and resamples to the target sample rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from .config import AudioIOConfig

logger = logging.getLogger("eit_pipeline.audio_io")


@dataclass
class AudioData:
    """Container for loaded audio."""
    waveform: np.ndarray   # 1-D float32 array
    sample_rate: int
    duration_s: float
    source_path: str
    original_sr: int


def load_audio(
    path: str,
    config: Optional[AudioIOConfig] = None,
    skip_seconds: float = 0.0,
) -> AudioData:
    """Load an audio file, resample, convert to mono, normalise.

    Parameters
    ----------
    path : str
        Path to the audio file (MP3, WAV, FLAC, etc.).
    config : AudioIOConfig, optional
        Audio loading parameters.  Uses defaults if not provided.
    skip_seconds : float
        Number of seconds to skip from the beginning (intro removal).

    Returns
    -------
    AudioData
        Loaded and preprocessed audio data.
    """
    if config is None:
        config = AudioIOConfig()

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    logger.info("Loading audio: %s", path)

    # Load with librosa (handles MP3, WAV, FLAC, etc.)
    waveform, original_sr = librosa.load(
        str(path_obj),
        sr=None,  # preserve original sample rate first
        mono=config.mono,
    )

    logger.info(
        "Loaded: sr=%d, duration=%.1fs, samples=%d",
        original_sr, len(waveform) / original_sr, len(waveform),
    )

    # Skip intro
    if skip_seconds > 0:
        skip_samples = int(skip_seconds * original_sr)
        if skip_samples >= len(waveform):
            raise ValueError(
                f"skip_seconds ({skip_seconds}) exceeds audio duration "
                f"({len(waveform) / original_sr:.1f}s)"
            )
        waveform = waveform[skip_samples:]
        logger.info("Skipped %.1fs intro, remaining: %.1fs",
                     skip_seconds, len(waveform) / original_sr)

    # Resample if needed
    target_sr = config.sample_rate
    if original_sr != target_sr:
        logger.info("Resampling %d -> %d Hz", original_sr, target_sr)
        waveform = librosa.resample(
            waveform, orig_sr=original_sr, target_sr=target_sr,
        )

    # Normalise amplitude
    waveform = _normalise_peak(waveform, config.normalise_peak_db)

    # Noise reduction (optional)
    if config.noise_reduce:
        waveform = _noise_reduce(
            waveform, target_sr,
            prop_decrease=config.noise_reduce_prop_decrease,
            stationary=config.noise_reduce_stationary,
        )

    duration = len(waveform) / target_sr
    logger.info("Audio ready: sr=%d, duration=%.1fs", target_sr, duration)

    return AudioData(
        waveform=waveform.astype(np.float32),
        sample_rate=target_sr,
        duration_s=duration,
        source_path=str(path_obj),
        original_sr=original_sr,
    )


def load_audio_segment(
    path: str,
    start_s: float,
    end_s: float,
    sample_rate: int = 16_000,
) -> Tuple[np.ndarray, int]:
    """Load a specific time range from an audio file.

    Parameters
    ----------
    path : str
        Path to audio file.
    start_s : float
        Start time in seconds.
    end_s : float
        End time in seconds.
    sample_rate : int
        Target sample rate.

    Returns
    -------
    tuple of (np.ndarray, int)
        Audio waveform and sample rate.
    """
    waveform, sr = librosa.load(
        path, sr=sample_rate, mono=True,
        offset=start_s, duration=end_s - start_s,
    )
    return waveform.astype(np.float32), sr


def save_audio(
    waveform: np.ndarray,
    sample_rate: int,
    path: str,
) -> None:
    """Save audio to WAV file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path_obj), waveform, sample_rate, subtype="PCM_16")
    logger.info("Saved audio: %s (%.1fs)", path, len(waveform) / sample_rate)


def _normalise_peak(waveform: np.ndarray, target_db: float) -> np.ndarray:
    """Normalise peak amplitude to target dBFS."""
    peak = np.max(np.abs(waveform))
    if peak < 1e-8:
        return waveform
    target_linear = 10 ** (target_db / 20.0)
    return waveform * (target_linear / peak)


def _noise_reduce(
    waveform: np.ndarray,
    sr: int,
    prop_decrease: float = 0.5,
    stationary: bool = True,
) -> np.ndarray:
    """Apply spectral-gating noise reduction."""
    try:
        import noisereduce as nr
        return nr.reduce_noise(
            y=waveform, sr=sr,
            prop_decrease=prop_decrease,
            stationary=stationary,
        ).astype(np.float32)
    except ImportError:
        logger.warning("noisereduce not installed; skipping noise reduction")
        return waveform
