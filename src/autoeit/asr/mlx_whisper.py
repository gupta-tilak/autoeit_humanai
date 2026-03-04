"""MLX-Whisper backend — Apple Silicon native inference via MLX framework.

Uses ``mlx-whisper`` which runs Whisper models natively on Apple Silicon
using the MLX framework with unified memory, giving near-GPU performance
without external accelerators.
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


class MLXWhisperBackend(ASRBackend):
    """MLX-Whisper backend for Apple Silicon.

    Leverages Apple's MLX framework for efficient on-device inference.
    Supports HuggingFace model repos from the mlx-community.
    """

    # Map model_size → default mlx-community repo
    MODEL_REPOS = {
        "tiny":   "mlx-community/whisper-tiny-mlx",
        "base":   "mlx-community/whisper-base-mlx",
        "small":  "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large":  "mlx-community/whisper-large-v3-mlx",
    }

    def __init__(self, config: ASRConfig):
        self.config = config
        self._loaded = False
        self._model_repo: str = ""

    def load_model(self) -> None:
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "mlx-whisper is not installed. Install with: pip install mlx-whisper"
            )

        # Resolve model repo
        if self.config.model_path:
            self._model_repo = self.config.model_path
        elif self.config.mlx_model_repo:
            self._model_repo = self.config.mlx_model_repo
        else:
            self._model_repo = self.MODEL_REPOS.get(
                self.config.model_size,
                self.MODEL_REPOS["small"],
            )

        logger.info("MLX-Whisper backend ready with model: %s", self._model_repo)
        # mlx_whisper downloads & caches automatically on first transcribe
        self._loaded = True

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

        import mlx_whisper

        t0 = time.time()

        # Build decode options
        # MLX-Whisper does not yet support beam search; use greedy decoding
        # beam_size must be None (not even 1) to avoid NotImplementedError
        decode_options = {
            "language": language,
            "temperature": temperature if temperature > 0 else 0,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "word_timestamps": self.config.word_timestamps,
        }
        if initial_prompt:
            decode_options["initial_prompt"] = initial_prompt

        # Filter out None values
        decode_options = {k: v for k, v in decode_options.items() if v is not None}

        raw = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self._model_repo,
            **decode_options,
        )

        elapsed = time.time() - t0
        return self._parse_result(raw, language, elapsed)

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
        """Transcribe from numpy array — write temp WAV then transcribe."""
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            audio_np = np.asarray(audio_array, dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            sf.write(tmp_path, audio_np, sample_rate, subtype="PCM_16")
            return self.transcribe(
                tmp_path,
                language=language,
                initial_prompt=initial_prompt,
                temperature=temperature,
                beam_size=beam_size,
                **kwargs,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @property
    def backend_name(self) -> str:
        return "MLX-Whisper"

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "backend": "mlx_whisper",
            "model_repo": self._model_repo,
            "model_size": self.config.model_size,
        }

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_result(raw: Dict[str, Any], language: str, elapsed: float) -> ASRResult:
        """Parse mlx_whisper.transcribe() output into ASRResult."""
        segments = []
        total_log_prob = 0.0
        total_no_speech = 0.0
        count = 0

        for seg in raw.get("segments", []):
            text = seg.get("text", "").strip()
            if not text:
                continue
            log_prob = seg.get("avg_logprob", 0.0)
            no_speech = seg.get("no_speech_prob", 0.0)

            words = []
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                    "probability": w.get("probability", 0.0),
                })

            segments.append(ASRSegment(
                text=text,
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                avg_log_prob=log_prob,
                no_speech_prob=no_speech,
                words=words,
            ))
            total_log_prob += log_prob
            total_no_speech += no_speech
            count += 1

        full_text = raw.get("text", "").strip()
        avg_lp = total_log_prob / max(count, 1)
        avg_ns = total_no_speech / max(count, 1)

        return ASRResult(
            text=full_text,
            segments=segments,
            language=language,
            avg_log_prob=avg_lp,
            no_speech_prob=avg_ns,
            processing_time_s=elapsed,
            metadata={"raw_segments_count": count},
        )
