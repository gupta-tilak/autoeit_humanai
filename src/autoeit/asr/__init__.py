"""ASR backend package — model-agnostic abstraction layer."""

from .base import ASRBackend, ASRSegment
from .factory import create_asr_backend

__all__ = ["ASRBackend", "ASRSegment", "create_asr_backend"]
