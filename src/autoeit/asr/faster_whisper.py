"""faster-whisper backend — CTranslate2-based CPU/GPU inference.

Uses ``faster-whisper`` which provides a CTranslate2 backend for Whisper.
Efficient on CPU with ~4x speedup over OpenAI Whisper.  Works well on
all platforms including Apple Silicon (CPU mode).
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import ASRConfig
from .base import ASRBackend, ASRResult, ASRSegment

logger = logging.getLogger(__name__)


class FasterWhisperBackend(ASRBackend):
    """faster-whisper (CTranslate2) backend."""

    def __init__(self, config: ASRConfig):
        self.config = config
        self._model = None
        self._loaded = False

    def load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed. Install with: pip install faster-whisper"
            )

        model_id = self.config.model_path or self.config.model_size
        logger.info("Loading faster-whisper model: %s", model_id)

        # On Apple Silicon, use CPU with int8
        device = "cpu"
        compute_type = "int8"

        self._model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
            cpu_threads=self.config.whisper_cpp_threads,
        )
        self._loaded = True
        logger.info("faster-whisper model loaded: %s (device=%s, compute=%s)", model_id, device, compute_type)

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
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        t0 = time.time()

        segments_gen, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            best_of=self.config.best_of,
            temperature=temperature,
            condition_on_previous_text=self.config.condition_on_previous_text,
            word_timestamps=self.config.word_timestamps,
            vad_filter=self.config.vad_filter,
            initial_prompt=initial_prompt or None,
        )

        segments = []
        full_text_parts = []
        total_log_prob = 0.0
        total_no_speech = 0.0
        count = 0

        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue

            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    })

            segments.append(ASRSegment(
                text=text,
                start=seg.start,
                end=seg.end,
                avg_log_prob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
                words=words,
            ))
            full_text_parts.append(text)
            total_log_prob += seg.avg_logprob
            total_no_speech += seg.no_speech_prob
            count += 1

        elapsed = time.time() - t0
        full_text = " ".join(full_text_parts)

        return ASRResult(
            text=full_text,
            segments=segments,
            language=language,
            avg_log_prob=total_log_prob / max(count, 1),
            no_speech_prob=total_no_speech / max(count, 1),
            processing_time_s=elapsed,
            metadata={
                "language_probability": info.language_probability if info else 0.0,
                "duration": info.duration if info else 0.0,
            },
        )

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
        """faster-whisper supports numpy arrays directly."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        audio_np = np.asarray(audio_array, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        t0 = time.time()

        segments_gen, info = self._model.transcribe(
            audio_np,
            language=language,
            beam_size=beam_size,
            best_of=self.config.best_of,
            temperature=temperature,
            condition_on_previous_text=self.config.condition_on_previous_text,
            word_timestamps=self.config.word_timestamps,
            vad_filter=self.config.vad_filter,
            initial_prompt=initial_prompt or None,
        )

        segments = []
        full_text_parts = []
        total_log_prob = 0.0
        total_no_speech = 0.0
        count = 0

        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue

            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    })

            segments.append(ASRSegment(
                text=text,
                start=seg.start,
                end=seg.end,
                avg_log_prob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
                words=words,
            ))
            full_text_parts.append(text)
            total_log_prob += seg.avg_logprob
            total_no_speech += seg.no_speech_prob
            count += 1

        elapsed = time.time() - t0
        full_text = " ".join(full_text_parts)

        return ASRResult(
            text=full_text,
            segments=segments,
            language=language,
            avg_log_prob=total_log_prob / max(count, 1),
            no_speech_prob=total_no_speech / max(count, 1),
            processing_time_s=elapsed,
            metadata={
                "language_probability": info.language_probability if info else 0.0,
                "duration": info.duration if info else 0.0,
            },
        )

    @property
    def backend_name(self) -> str:
        return "faster-whisper"

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "backend": "faster_whisper",
            "model_size": self.config.model_size,
            "model_path": self.config.model_path or "",
        }

    def is_loaded(self) -> bool:
        return self._loaded
