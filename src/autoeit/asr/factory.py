"""Factory for creating ASR backend instances.

One-liner to switch between whisper.cpp, MLX-Whisper, faster-whisper,
or any future custom backend.
"""

from __future__ import annotations

import logging
from typing import Dict, Type

from ..config import ASRConfig
from .base import ASRBackend

logger = logging.getLogger(__name__)

# Registry of available backends
_BACKENDS: Dict[str, str] = {
    "whisper_cpp":     "autoeit.asr.whisper_cpp.WhisperCppBackend",
    "mlx_whisper":     "autoeit.asr.mlx_whisper.MLXWhisperBackend",
    "faster_whisper":  "autoeit.asr.faster_whisper.FasterWhisperBackend",
}


def register_backend(name: str, class_path: str) -> None:
    """Register a custom ASR backend.

    Parameters
    ----------
    name : str
        Short identifier (used in config.asr.backend).
    class_path : str
        Fully-qualified class name, e.g. ``"mypackage.my_backend.MyBackend"``.
    """
    _BACKENDS[name] = class_path
    logger.info("Registered custom ASR backend: %s -> %s", name, class_path)


def create_asr_backend(config: ASRConfig) -> ASRBackend:
    """Instantiate and return the appropriate ASR backend.

    Parameters
    ----------
    config : ASRConfig
        Configuration including ``backend`` field.

    Returns
    -------
    ASRBackend
        An uninitialised backend — call ``load_model()`` to activate.
    """
    backend_name = config.backend

    if backend_name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(
            f"Unknown ASR backend '{backend_name}'. Available: {available}"
        )

    class_path = _BACKENDS[backend_name]
    cls = _import_class(class_path)

    logger.info("Creating ASR backend: %s (%s)", backend_name, class_path)
    return cls(config)


def _import_class(dotted_path: str) -> Type:
    """Import a class from a dotted module path."""
    module_path, _, class_name = dotted_path.rpartition(".")
    import importlib
    module = importlib.import_module(f".{module_path.split('.')[-1]}", package="autoeit.asr")
    return getattr(module, class_name)


def list_backends() -> Dict[str, str]:
    """Return all registered backends."""
    return dict(_BACKENDS)
