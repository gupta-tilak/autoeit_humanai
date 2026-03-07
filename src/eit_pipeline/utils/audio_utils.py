"""Audio utility functions."""

from __future__ import annotations

import numpy as np


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS energy of audio."""
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_rms_db(audio: np.ndarray) -> float:
    """Compute RMS energy in dBFS."""
    rms = compute_rms(audio)
    if rms < 1e-10:
        return -100.0
    return float(20 * np.log10(rms))


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to H:MM:SS.ss format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"
