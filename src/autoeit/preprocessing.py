"""Audio preprocessing module.

Handles:
  - MP3 → WAV conversion (16kHz mono 16-bit PCM)
  - Intro skipping (2:30 for most files, 12:00 for 038012)
  - Peak normalisation
  - Light noise reduction (spectral gating)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import AudioFileConfig, PreprocessingConfig

logger = logging.getLogger(__name__)


def preprocess_audio(
    audio_path: str,
    file_config: AudioFileConfig,
    preproc_config: PreprocessingConfig,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, int, str]:
    """Full preprocessing pipeline for one audio file.

    Returns
    -------
    audio : np.ndarray
        Float32 mono audio array (post-processing).
    sample_rate : int
        Sample rate (always ``preproc_config.sample_rate``).
    wav_path : str
        Path to the cached WAV file on disk.
    """
    import librosa
    import soundfile as sf

    sr = preproc_config.sample_rate

    # ------------------------------------------------------------------
    # 1. Load audio
    # ------------------------------------------------------------------
    logger.info(
        "Loading %s (skip first %.0fs)",
        file_config.filename, file_config.skip_seconds,
    )

    # Load full audio first, then trim
    audio, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
    logger.info(
        "Loaded: %.1fs @ %dHz (%d samples)",
        len(audio) / sr, sr, len(audio),
    )

    # ------------------------------------------------------------------
    # 2. Skip intro
    # ------------------------------------------------------------------
    skip_samples = int(file_config.skip_seconds * sr)
    if skip_samples >= len(audio):
        raise ValueError(
            f"Skip offset ({file_config.skip_seconds}s = {skip_samples} samples) "
            f"exceeds audio length ({len(audio)} samples = {len(audio)/sr:.1f}s)"
        )
    audio = audio[skip_samples:]
    logger.info(
        "After intro skip: %.1fs (%d samples)",
        len(audio) / sr, len(audio),
    )

    # ------------------------------------------------------------------
    # 3. Peak normalisation
    # ------------------------------------------------------------------
    audio = normalise_peak(audio, preproc_config.normalise_peak_db)

    # ------------------------------------------------------------------
    # 4. Noise reduction
    # ------------------------------------------------------------------
    if preproc_config.noise_reduce:
        audio = reduce_noise(
            audio, sr,
            prop_decrease=preproc_config.noise_reduce_prop_decrease,
            stationary=preproc_config.noise_reduce_stationary,
        )

    # ------------------------------------------------------------------
    # 5. Save to WAV cache
    # ------------------------------------------------------------------
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        wav_path = str(
            Path(cache_dir) / f"{file_config.participant_id}_{file_config.eit_version}_preprocessed.wav"
        )
    else:
        wav_path = tempfile.mktemp(suffix=".wav")

    sf.write(wav_path, audio, sr, subtype="PCM_16")
    logger.info("Saved preprocessed audio: %s (%.1fs)", wav_path, len(audio) / sr)

    return audio, sr, wav_path


# ---------------------------------------------------------------------------
# Individual steps (exposed for notebook use / analysis)
# ---------------------------------------------------------------------------

def normalise_peak(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Peak-normalise audio to *target_db* dBFS."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        logger.warning("Audio is near-silent — skipping normalisation")
        return audio

    target_linear = 10 ** (target_db / 20.0)
    gain = target_linear / peak
    normalised = audio * gain

    logger.debug(
        "Peak normalisation: peak=%.4f → target=%.4f (gain=%.2f)",
        peak, target_linear, gain,
    )
    return normalised


def reduce_noise(
    audio: np.ndarray,
    sr: int,
    prop_decrease: float = 0.5,
    stationary: bool = True,
) -> np.ndarray:
    """Light spectral-gating noise reduction.

    Uses ``noisereduce`` with conservative settings to help ASR
    without distorting non-native speech patterns.
    """
    try:
        import noisereduce as nr

        logger.info(
            "Applying noise reduction (prop_decrease=%.2f, stationary=%s)",
            prop_decrease, stationary,
        )
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=prop_decrease,
            stationary=stationary,
        )
        return reduced.astype(np.float32)

    except ImportError:
        logger.warning("noisereduce not installed — skipping noise reduction")
        return audio


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS energy of an audio array."""
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_duration(audio: np.ndarray, sr: int) -> float:
    """Duration in seconds."""
    return len(audio) / sr
