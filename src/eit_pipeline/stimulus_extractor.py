"""Stimulus Extractor — Bootstrap tool to extract reference stimulus audio.

Since separate stimulus audio files are not available, this tool extracts
stimulus segments from a reference recording using full-file ASR with
word-level timestamps and fuzzy text matching against known stimulus sentences.

Once extracted, the stimulus audio files serve as reference for the
stimulus alignment stage across all recordings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from .audio_io import AudioData, load_audio, save_audio
from .config import TARGET_SENTENCES, AudioIOConfig

logger = logging.getLogger("eit_pipeline.stimulus_extractor")


@dataclass
class ExtractedStimulus:
    """A single extracted stimulus segment."""
    sentence_id: int
    sentence_text: str
    start_s: float
    end_s: float
    confidence: float
    source_file: str


def extract_stimuli_from_recording(
    audio_path: str,
    output_dir: str,
    skip_seconds: float = 150.0,
    sample_rate: int = 16_000,
    sentences: Optional[List[str]] = None,
    padding_before_s: float = 0.2,
    padding_after_s: float = 0.3,
) -> List[ExtractedStimulus]:
    """Extract stimulus audio segments from a reference recording using ASR.

    This function:
    1. Loads the full recording
    2. Runs full-file ASR to get word-level timestamps
    3. Fuzzy-matches ASR output against known stimulus sentences
    4. Extracts and saves each stimulus as a separate WAV file

    Parameters
    ----------
    audio_path : str
        Path to the reference EIT recording.
    output_dir : str
        Directory to save extracted stimulus WAV files.
    skip_seconds : float
        Seconds to skip from the beginning.
    sample_rate : int
        Target sample rate.
    sentences : list of str, optional
        Stimulus sentences.  Defaults to TARGET_SENTENCES.
    padding_before_s : float
        Seconds of padding before stimulus start.
    padding_after_s : float
        Seconds of padding after stimulus end.

    Returns
    -------
    list of ExtractedStimulus
        Extracted stimulus metadata.
    """
    if sentences is None:
        sentences = TARGET_SENTENCES

    # Load audio
    config = AudioIOConfig(sample_rate=sample_rate, noise_reduce=False)
    audio = load_audio(audio_path, config=config, skip_seconds=skip_seconds)

    # Run full-file ASR for word-level timestamps
    logger.info("Running full-file ASR for stimulus extraction...")
    words = _run_full_file_asr(audio)

    if not words:
        raise RuntimeError("ASR returned no words — cannot extract stimuli")

    logger.info("ASR returned %d words", len(words))

    # Match stimulus sentences against ASR word-level output
    matches = _match_stimuli_to_words(sentences, words)

    # Extract and save audio segments
    extracted = []
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for match in matches:
        sid = match["sentence_id"]
        start = max(0, match["start_s"] - padding_before_s)
        end = min(audio.duration_s, match["end_s"] + padding_after_s)

        start_sample = int(start * audio.sample_rate)
        end_sample = int(end * audio.sample_rate)
        segment_audio = audio.waveform[start_sample:end_sample]

        filename = f"stimulus_{sid:02d}.wav"
        save_audio(segment_audio, audio.sample_rate, str(out_path / filename))

        stimulus = ExtractedStimulus(
            sentence_id=sid,
            sentence_text=sentences[sid - 1],
            start_s=start,
            end_s=end,
            confidence=match["confidence"],
            source_file=audio_path,
        )
        extracted.append(stimulus)
        logger.info(
            "Extracted stimulus %d: %.1f-%.1fs (conf=%.2f) '%s'",
            sid, start, end, match["confidence"], sentences[sid - 1][:40],
        )

    # Save metadata
    meta_path = out_path / "extraction_metadata.json"
    meta = {
        "source_file": audio_path,
        "skip_seconds": skip_seconds,
        "sample_rate": sample_rate,
        "num_stimuli": len(extracted),
        "stimuli": [asdict(s) for s in extracted],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Extracted %d / %d stimuli to %s", len(extracted), len(sentences), output_dir)
    return extracted


def _run_full_file_asr(audio: AudioData) -> List[Dict]:
    """Run ASR on full audio and return word-level timestamps.

    Returns a list of dicts: [{"word": str, "start": float, "end": float}, ...]
    """
    try:
        import mlx_whisper
    except ImportError:
        raise ImportError(
            "mlx_whisper is required for stimulus extraction. "
            "Install with: pip install mlx-whisper"
        )

    # Save to temp WAV for ASR
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    save_audio(audio.waveform, audio.sample_rate, tmp_path)

    try:
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo="mlx-community/whisper-small-mlx",
            language="es",
            word_timestamps=True,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=(
                "Transcripción de oraciones en español con pausas entre ellas. "
                "Quiero cortarme el pelo. El libro está en la mesa."
            ),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Extract word-level timestamps
    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append({
                "word": w.get("word", "").strip(),
                "start": w.get("start", 0.0),
                "end": w.get("end", 0.0),
            })

    return words


def _match_stimuli_to_words(
    sentences: List[str],
    words: List[Dict],
) -> List[Dict]:
    """Fuzzy-match stimulus sentences against ASR word timestamps.

    Uses a sliding window approach over the word sequence to find the best
    match for each stimulus sentence in order.
    """
    from difflib import SequenceMatcher

    matches = []
    word_idx = 0  # Track position to enforce ordering

    for sid, sentence in enumerate(sentences, start=1):
        sentence_words = sentence.lower().replace("¿", "").replace("?", "").split()
        n_sentence_words = len(sentence_words)
        sentence_text = " ".join(sentence_words)

        best_score = 0.0
        best_start_idx = -1
        best_end_idx = -1

        # Sliding window over ASR words, starting from last match position
        search_start = max(0, word_idx - 5)
        for i in range(search_start, len(words) - n_sentence_words + 1):
            # Try windows of varying sizes around expected word count
            for window_size in range(
                max(1, n_sentence_words - 2),
                min(n_sentence_words + 4, len(words) - i + 1),
            ):
                window_words = [w["word"].lower().strip(".,;:!?¿¡") for w in words[i:i + window_size]]
                window_text = " ".join(window_words)

                score = SequenceMatcher(None, sentence_text, window_text).ratio()

                if score > best_score:
                    best_score = score
                    best_start_idx = i
                    best_end_idx = i + window_size - 1

        if best_score >= 0.4 and best_start_idx >= 0:
            matches.append({
                "sentence_id": sid,
                "start_s": words[best_start_idx]["start"],
                "end_s": words[best_end_idx]["end"],
                "confidence": best_score,
            })
            word_idx = best_end_idx + 1
            logger.debug(
                "Matched stimulus %d (score=%.2f): %.1f-%.1fs",
                sid, best_score,
                words[best_start_idx]["start"],
                words[best_end_idx]["end"],
            )
        else:
            logger.warning("Could not match stimulus %d: '%s'", sid, sentence[:40])
            # Still advance word_idx to maintain ordering
            word_idx = max(word_idx, best_end_idx + 1 if best_end_idx >= 0 else word_idx)

    return matches
