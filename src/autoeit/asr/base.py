"""Abstract base class for all ASR backends.

Every backend (whisper.cpp, MLX-Whisper, faster-whisper, fine-tuned models,
etc.) implements this interface.  The pipeline only talks to this abstraction
so swapping backends is a one-line config change.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ASRSegment:
    """A single ASR output segment (word or phrase level)."""
    text: str
    start: float          # seconds
    end: float            # seconds
    avg_log_prob: float = 0.0
    no_speech_prob: float = 0.0
    # Word-level timestamps (if available)
    words: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ASRResult:
    """Complete result from transcribing one audio segment."""
    text: str                              # full transcription text
    segments: List[ASRSegment] = field(default_factory=list)
    language: str = "es"
    avg_log_prob: float = 0.0              # average across all segments
    no_speech_prob: float = 0.0            # average across all segments
    processing_time_s: float = 0.0
    # Backend-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ASRBackend(abc.ABC):
    """Abstract ASR backend interface.

    Implementations must provide:
      - ``load_model()`` — load/initialize the model
      - ``transcribe()`` — transcribe a single audio file/array
      - ``transcribe_segment()`` — transcribe a numpy audio array
      - ``backend_name`` property
    """

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load the ASR model into memory."""
        ...

    @abc.abstractmethod
    def transcribe(
        self,
        audio_path: str,
        *,
        language: str = "es",
        initial_prompt: str = "",
        temperature: float = 0.0,
        beam_size: int = 5,
        **kwargs: Any,
    ) -> ASRResult:
        """Transcribe an audio file on disk.

        Parameters
        ----------
        audio_path : str
            Path to a WAV/MP3 file.
        language : str
            ISO 639-1 language code.
        initial_prompt : str
            Conditioning prompt for the decoder.
        temperature : float
            Sampling temperature (0 = greedy).
        beam_size : int
            Beam search width.

        Returns
        -------
        ASRResult
        """
        ...

    @abc.abstractmethod
    def transcribe_array(
        self,
        audio_array: Any,
        sample_rate: int = 16000,
        *,
        language: str = "es",
        initial_prompt: str = "",
        temperature: float = 0.0,
        beam_size: int = 5,
        **kwargs: Any,
    ) -> ASRResult:
        """Transcribe a numpy audio array directly.

        Parameters
        ----------
        audio_array : np.ndarray
            Audio samples (float32, mono).
        sample_rate : int
            Sample rate of the array.

        Returns
        -------
        ASRResult
        """
        ...

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Human-readable name of this backend."""
        ...

    @property
    @abc.abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Summary of loaded model (name, size, path, etc.)."""
        ...

    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return False
