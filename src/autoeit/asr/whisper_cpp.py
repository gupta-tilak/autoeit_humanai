"""whisper.cpp backend — Mac-native Metal-accelerated inference.

Uses the ``pywhispercpp`` Python bindings for whisper.cpp, giving us
Metal GPU acceleration on Apple Silicon with minimal memory footprint.
Falls back to the whisper.cpp CLI if bindings are unavailable.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import ASRConfig
from .base import ASRBackend, ASRResult, ASRSegment

logger = logging.getLogger(__name__)


class WhisperCppBackend(ASRBackend):
    """whisper.cpp backend with Metal acceleration on macOS.

    Supports two modes:
    1. **pywhispercpp** Python bindings (preferred)
    2. **CLI fallback** — calls the whisper.cpp ``main`` binary directly

    Model files are GGML format (``.bin``).
    """

    def __init__(self, config: ASRConfig):
        self.config = config
        self._model = None
        self._model_path: Optional[str] = None
        self._use_cli = False
        self._loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        model_path = self._resolve_model_path()
        self._model_path = model_path

        # Try pywhispercpp first
        try:
            from pywhispercpp.model import Model as WhisperCppModel

            logger.info("Loading whisper.cpp model via pywhispercpp: %s", model_path)
            self._model = WhisperCppModel(
                model=model_path,
                n_threads=self.config.whisper_cpp_threads,
            )
            self._use_cli = False
            self._loaded = True
            logger.info("whisper.cpp model loaded (pywhispercpp, Metal: %s)", self.config.whisper_cpp_use_gpu)
            return

        except ImportError:
            logger.warning("pywhispercpp not installed — will try CLI fallback")
        except Exception as e:
            logger.warning("pywhispercpp failed to load model: %s — trying CLI", e)

        # CLI fallback
        cli_bin = self.config.whisper_cpp_bin or self._find_whisper_cpp_binary()
        if cli_bin and Path(cli_bin).exists():
            logger.info("Using whisper.cpp CLI at: %s", cli_bin)
            self.config.whisper_cpp_bin = cli_bin
            self._use_cli = True
            self._loaded = True
        else:
            raise RuntimeError(
                "whisper.cpp backend: neither pywhispercpp nor CLI binary found. "
                "Install pywhispercpp (`pip install pywhispercpp`) or set "
                "config.asr.whisper_cpp_bin to the path of the whisper.cpp main binary."
            )

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

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

        if self._use_cli:
            result = self._transcribe_cli(audio_path, language, initial_prompt, temperature, beam_size)
        else:
            result = self._transcribe_bindings(audio_path, language, initial_prompt, temperature, beam_size)

        result.processing_time_s = time.time() - t0
        return result

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
        """Transcribe from numpy array by writing a temp WAV file."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        mode = "CLI" if self._use_cli else "pywhispercpp"
        return f"whisper.cpp ({mode})"

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "backend": "whisper_cpp",
            "mode": "cli" if self._use_cli else "pywhispercpp",
            "model_path": self._model_path or "",
            "model_size": self.config.model_size,
            "use_gpu": self.config.whisper_cpp_use_gpu,
            "threads": self.config.whisper_cpp_threads,
        }

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # pywhispercpp path
    # ------------------------------------------------------------------

    def _transcribe_bindings(
        self, audio_path: str, language: str, initial_prompt: str,
        temperature: float, beam_size: int,
    ) -> ASRResult:
        """Transcribe using pywhispercpp Python bindings."""
        segments_raw = self._model.transcribe(
            media=audio_path,
            language=language,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            temperature=temperature,
            word_timestamps=self.config.word_timestamps,
        )

        segments = []
        full_text_parts = []

        for seg in segments_raw:
            text = seg.text.strip() if hasattr(seg, "text") else str(seg).strip()
            start = seg.t0 / 100.0 if hasattr(seg, "t0") else 0.0
            end = seg.t1 / 100.0 if hasattr(seg, "t1") else 0.0

            if text:
                full_text_parts.append(text)
                segments.append(ASRSegment(
                    text=text, start=start, end=end,
                    avg_log_prob=0.0,  # pywhispercpp doesn't expose this
                    no_speech_prob=0.0,
                ))

        full_text = " ".join(full_text_parts)
        return ASRResult(
            text=full_text,
            segments=segments,
            language=language,
        )

    # ------------------------------------------------------------------
    # CLI fallback path
    # ------------------------------------------------------------------

    def _transcribe_cli(
        self, audio_path: str, language: str, initial_prompt: str,
        temperature: float, beam_size: int,
    ) -> ASRResult:
        """Transcribe using whisper.cpp CLI binary."""
        cmd = [
            self.config.whisper_cpp_bin,
            "-m", self._model_path,
            "-f", audio_path,
            "-l", language,
            "--beam-size", str(beam_size),
            "-t", str(self.config.whisper_cpp_threads),
            "--output-txt",
            "--no-prints",
        ]

        if initial_prompt:
            cmd.extend(["--prompt", initial_prompt])

        if not self.config.whisper_cpp_use_gpu:
            cmd.append("--no-gpu")

        logger.debug("Running whisper.cpp CLI: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error("whisper.cpp CLI error: %s", result.stderr)
            raise RuntimeError(f"whisper.cpp failed: {result.stderr}")

        text = result.stdout.strip()
        return ASRResult(text=text, language=language)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_model_path(self) -> str:
        """Find the GGML model file."""
        # Explicit path takes priority
        if self.config.whisper_cpp_model_path and Path(self.config.whisper_cpp_model_path).exists():
            return self.config.whisper_cpp_model_path

        if self.config.model_path and Path(self.config.model_path).exists():
            return self.config.model_path

        # Check common locations
        model_name = f"ggml-{self.config.model_size}.bin"
        search_dirs = [
            Path.home() / ".cache" / "whisper.cpp",
            Path.home() / ".cache" / "whisper",
            Path.home() / "whisper.cpp" / "models",
            Path("/usr/local/share/whisper.cpp/models"),
            Path.cwd() / "models",
        ]

        for d in search_dirs:
            candidate = d / model_name
            if candidate.exists():
                return str(candidate)

        # Try to download via pywhispercpp helper
        try:
            from pywhispercpp.utils import download_model
            logger.info("Downloading whisper.cpp model: %s", self.config.model_size)
            model_dir = Path.home() / ".cache" / "whisper.cpp"
            model_dir.mkdir(parents=True, exist_ok=True)
            path = download_model(self.config.model_size, str(model_dir))
            return str(path)
        except Exception as e:
            logger.debug("Could not auto-download model: %s", e)

        raise FileNotFoundError(
            f"whisper.cpp model '{model_name}' not found. "
            f"Set config.asr.whisper_cpp_model_path or download it manually. "
            f"Searched: {[str(d) for d in search_dirs]}"
        )

    @staticmethod
    def _find_whisper_cpp_binary() -> Optional[str]:
        """Try to locate the whisper.cpp CLI binary."""
        names = ["whisper-cpp", "whisper", "main"]
        for name in names:
            try:
                result = subprocess.run(
                    ["which", name], capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                pass
        # Check common build locations
        common = [
            Path.home() / "whisper.cpp" / "main",
            Path.home() / "whisper.cpp" / "build" / "bin" / "main",
            Path("/usr/local/bin/whisper-cpp"),
        ]
        for p in common:
            if p.exists():
                return str(p)
        return None
