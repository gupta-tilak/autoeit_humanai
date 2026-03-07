"""Stage 7 — ASR Transcription.

Transcribes only response segments using the configured ASR backend.
Supports batched inference and multi-pass transcription.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .audio_io import AudioData, save_audio
from .config import ASRConfig
from .response_window import ResponseSegment

logger = logging.getLogger("eit_pipeline.asr_transcription")


@dataclass
class TranscriptionResult:
    """Result from transcribing a single response segment."""
    sentence_id: int
    response_start_s: float
    response_end_s: float
    transcript: str
    raw_transcript: str
    avg_log_prob: float = 0.0
    no_speech_prob: float = 0.0
    processing_time_s: float = 0.0
    pass_number: int = 1
    word_timestamps: List[Dict[str, Any]] = field(default_factory=list)


def transcribe_responses(
    responses: List[ResponseSegment],
    recording: AudioData,
    config: Optional[ASRConfig] = None,
) -> List[TranscriptionResult]:
    """Transcribe all response segments.

    Parameters
    ----------
    responses : list of ResponseSegment
        Response windows to transcribe.
    recording : AudioData
        Full recording audio data.
    config : ASRConfig, optional
        ASR configuration.

    Returns
    -------
    list of TranscriptionResult
        Transcription results, one per response segment.
    """
    if config is None:
        config = ASRConfig()

    logger.info(
        "Transcribing %d responses (backend=%s, model=%s)",
        len(responses), config.backend, config.model_size,
    )

    # Load ASR model
    asr_model = _load_asr_model(config)

    results = []
    for resp in responses:
        if resp.response_end_s == 0.0:
            # No response detected
            results.append(TranscriptionResult(
                sentence_id=resp.sentence_id,
                response_start_s=0.0,
                response_end_s=0.0,
                transcript="[no response]",
                raw_transcript="",
            ))
            continue

        # Extract audio segment
        start_sample = int(resp.response_start_s * recording.sample_rate)
        end_sample = int(resp.response_end_s * recording.sample_rate)
        segment_audio = recording.waveform[start_sample:end_sample]

        if len(segment_audio) == 0:
            results.append(TranscriptionResult(
                sentence_id=resp.sentence_id,
                response_start_s=resp.response_start_s,
                response_end_s=resp.response_end_s,
                transcript="[no response]",
                raw_transcript="",
            ))
            continue

        # First pass: greedy (temperature=0)
        t0 = time.time()
        result = _transcribe_segment(
            segment_audio, recording.sample_rate, asr_model, config,
            temperature=config.temperature,
        )
        elapsed = time.time() - t0

        transcription = TranscriptionResult(
            sentence_id=resp.sentence_id,
            response_start_s=resp.response_start_s,
            response_end_s=resp.response_end_s,
            transcript=result["text"].strip(),
            raw_transcript=result["text"].strip(),
            avg_log_prob=result.get("avg_log_prob", 0.0),
            no_speech_prob=result.get("no_speech_prob", 0.0),
            processing_time_s=elapsed,
            pass_number=1,
            word_timestamps=result.get("words", []),
        )

        # Multi-pass: retry with higher temperature if confidence is low
        if (
            config.multi_pass
            and transcription.avg_log_prob < config.low_confidence_threshold
        ):
            logger.debug(
                "Low confidence for stimulus %d (%.2f < %.2f) — retrying",
                resp.sentence_id,
                transcription.avg_log_prob,
                config.low_confidence_threshold,
            )
            t0 = time.time()
            retry_result = _transcribe_segment(
                segment_audio, recording.sample_rate, asr_model, config,
                temperature=config.second_pass_temperature,
            )
            elapsed2 = time.time() - t0

            # Use retry if it has better confidence
            retry_prob = retry_result.get("avg_log_prob", -999)
            if retry_prob > transcription.avg_log_prob:
                transcription.transcript = retry_result["text"].strip()
                transcription.raw_transcript = retry_result["text"].strip()
                transcription.avg_log_prob = retry_prob
                transcription.no_speech_prob = retry_result.get("no_speech_prob", 0.0)
                transcription.processing_time_s += elapsed2
                transcription.pass_number = 2
                transcription.word_timestamps = retry_result.get("words", [])

        logger.debug(
            "Stimulus %d: '%s' (prob=%.2f, pass=%d)",
            resp.sentence_id,
            transcription.transcript[:60],
            transcription.avg_log_prob,
            transcription.pass_number,
        )
        results.append(transcription)

    logger.info(
        "Transcribed %d segments (%d with text)",
        len(results),
        sum(1 for r in results if r.transcript != "[no response]"),
    )
    return results


def _load_asr_model(config: ASRConfig) -> Any:
    """Load the ASR model based on configuration."""
    if config.backend == "mlx_whisper":
        # MLX-Whisper doesn't need explicit model loading — it loads on first call
        return {"backend": "mlx_whisper", "repo": config.mlx_model_repo}

    elif config.backend == "faster_whisper":
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required. Install: pip install faster-whisper"
            )
        model = WhisperModel(
            config.model_size,
            device="cpu",
            compute_type="int8",
        )
        return {"backend": "faster_whisper", "model": model}

    else:
        raise ValueError(
            f"ASR backend '{config.backend}' is not supported. "
            f"Supported backends: 'mlx_whisper', 'faster_whisper'."
        )


def _transcribe_segment(
    audio: np.ndarray,
    sr: int,
    model: Dict,
    config: ASRConfig,
    temperature: float = 0.0,
) -> Dict:
    """Transcribe a single audio segment."""
    if model["backend"] == "mlx_whisper":
        return _transcribe_mlx(audio, sr, model["repo"], config, temperature)
    elif model["backend"] == "faster_whisper":
        return _transcribe_faster_whisper(audio, sr, model["model"], config, temperature)
    else:
        raise ValueError(f"Unknown ASR backend: {model['backend']}")


def _transcribe_mlx(
    audio: np.ndarray,
    sr: int,
    model_repo: str,
    config: ASRConfig,
    temperature: float,
) -> Dict:
    """Transcribe using MLX-Whisper."""
    try:
        import mlx_whisper
    except ImportError:
        raise ImportError("mlx-whisper required. Install: pip install mlx-whisper")

    # Save to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    save_audio(audio, sr, tmp_path)

    try:
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=model_repo,
            language=config.language,
            word_timestamps=config.word_timestamps,
            temperature=temperature,
            condition_on_previous_text=config.condition_on_previous_text,
            initial_prompt=config.initial_prompt,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Extract results
    text = result.get("text", "").strip()
    segments = result.get("segments", [])

    avg_log_prob = 0.0
    no_speech_prob = 0.0
    words = []

    if segments:
        avg_log_prob = np.mean([s.get("avg_logprob", 0.0) for s in segments])
        no_speech_prob = np.mean([s.get("no_speech_prob", 0.0) for s in segments])
        for seg in segments:
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                })

    return {
        "text": text,
        "avg_log_prob": float(avg_log_prob),
        "no_speech_prob": float(no_speech_prob),
        "words": words,
    }


def _transcribe_faster_whisper(
    audio: np.ndarray,
    sr: int,
    model: Any,
    config: ASRConfig,
    temperature: float,
) -> Dict:
    """Transcribe using faster-whisper."""
    segments_gen, info = model.transcribe(
        audio,
        language=config.language,
        beam_size=config.beam_size,
        best_of=config.best_of,
        temperature=temperature,
        word_timestamps=config.word_timestamps,
        condition_on_previous_text=config.condition_on_previous_text,
        initial_prompt=config.initial_prompt,
        vad_filter=config.vad_filter,
    )

    segments = list(segments_gen)
    text = " ".join(s.text.strip() for s in segments)

    avg_log_prob = 0.0
    no_speech_prob = 0.0
    words = []

    if segments:
        avg_log_prob = np.mean([s.avg_logprob for s in segments])
        no_speech_prob = np.mean([s.no_speech_prob for s in segments])
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                    })

    return {
        "text": text,
        "avg_log_prob": float(avg_log_prob),
        "no_speech_prob": float(no_speech_prob),
        "words": words,
    }
