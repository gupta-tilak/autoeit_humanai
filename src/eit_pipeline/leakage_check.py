"""Stage 8 — Stimulus Leakage Check.

Ensures stimulus audio was not mistakenly transcribed as a learner response.
Compares each transcript against its corresponding stimulus sentence text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from .asr_transcription import TranscriptionResult
from .config import TARGET_SENTENCES, LeakageCheckConfig

logger = logging.getLogger("eit_pipeline.leakage_check")


@dataclass
class LeakageFlag:
    """A flagged potential stimulus leakage."""
    sentence_id: int
    transcript: str
    stimulus_text: str
    similarity_score: float
    flagged: bool
    reason: str
    # True when similarity >= discard_threshold: segment should be cleared
    # (replaced with "[stimulus echo \u2014 discarded]") in the final output.
    should_discard: bool = False


def check_leakage(
    transcriptions: List[TranscriptionResult],
    config: Optional[LeakageCheckConfig] = None,
    stimulus_sentences: Optional[List[str]] = None,
) -> List[LeakageFlag]:
    """Check transcriptions for stimulus leakage.

    Parameters
    ----------
    transcriptions : list of TranscriptionResult
        ASR transcription results.
    config : LeakageCheckConfig, optional
        Leakage check parameters.
    stimulus_sentences : list of str, optional
        Reference stimulus texts. Defaults to TARGET_SENTENCES.

    Returns
    -------
    list of LeakageFlag
        Leakage analysis results for each transcription.
    """
    if config is None:
        config = LeakageCheckConfig()

    if not config.enabled:
        logger.info("Leakage check disabled")
        return []

    if stimulus_sentences is None:
        stimulus_sentences = TARGET_SENTENCES

    logger.info("Running stimulus leakage check (%s)", config.similarity_method)

    flags = []
    for trans in transcriptions:
        if trans.transcript in ("[no response]", ""):
            flags.append(LeakageFlag(
                sentence_id=trans.sentence_id,
                transcript=trans.transcript,
                stimulus_text="",
                similarity_score=0.0,
                flagged=False,
                reason="",
            ))
            continue

        # Get corresponding stimulus text
        sid = trans.sentence_id
        if sid < 1 or sid > len(stimulus_sentences):
            logger.warning("Sentence ID %d out of range", sid)
            continue

        stimulus_text = stimulus_sentences[sid - 1]

        # Compute similarity
        similarity = _compute_similarity(
            trans.transcript, stimulus_text, config.similarity_method,
        )

        discard_threshold = getattr(config, "discard_threshold", 0.90)
        should_discard = similarity >= discard_threshold
        flagged = similarity >= config.similarity_threshold
        reason = ""
        if should_discard:
            reason = f"stimulus_echo_discard(sim={similarity:.2f})"
            logger.warning(
                "Stimulus echo DISCARDED for sentence %d: sim=%.2f '%s'",
                sid, similarity, trans.transcript[:50],
            )
        elif flagged:
            reason = f"stimulus_echo(sim={similarity:.2f})"
            logger.warning(
                "Stimulus leakage flagged for sentence %d: sim=%.2f '%s'",
                sid, similarity, trans.transcript[:50],
            )

        flags.append(LeakageFlag(
            sentence_id=sid,
            transcript=trans.transcript,
            stimulus_text=stimulus_text,
            similarity_score=similarity,
            flagged=flagged,
            reason=reason,
            should_discard=should_discard,
        ))

    flagged_count = sum(1 for f in flags if f.flagged)
    logger.info(
        "Leakage check: %d / %d flagged",
        flagged_count, len(flags),
    )
    return flags


def _compute_similarity(
    transcript: str,
    stimulus: str,
    method: str = "rapidfuzz",
) -> float:
    """Compute text similarity between transcript and stimulus.

    Parameters
    ----------
    transcript : str
        ASR transcript.
    stimulus : str
        Stimulus sentence text.
    method : str
        Similarity method: "rapidfuzz" or "jiwer".

    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    # Normalise texts
    t = _normalise_text(transcript)
    s = _normalise_text(stimulus)

    if not t or not s:
        return 0.0

    if method == "rapidfuzz":
        return _similarity_rapidfuzz(t, s)
    elif method == "jiwer":
        return _similarity_jiwer(t, s)
    else:
        # Fallback to simple Jaccard
        return _similarity_jaccard(t, s)


def _normalise_text(text: str) -> str:
    """Normalise text for comparison."""
    text = text.lower().strip()
    # Remove punctuation
    for ch in ".,;:!?¿¡\"'()[]{}":
        text = text.replace(ch, "")
    # Collapse whitespace
    return " ".join(text.split())


def _similarity_rapidfuzz(a: str, b: str) -> float:
    """Compute similarity using rapidfuzz."""
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b) / 100.0
    except ImportError:
        logger.warning("rapidfuzz not installed — falling back to Jaccard")
        return _similarity_jaccard(a, b)


def _similarity_jiwer(a: str, b: str) -> float:
    """Compute similarity using jiwer (1 - WER)."""
    try:
        import jiwer
        wer = jiwer.wer(b, a)
        return max(0.0, 1.0 - wer)
    except ImportError:
        logger.warning("jiwer not installed — falling back to Jaccard")
        return _similarity_jaccard(a, b)


def _similarity_jaccard(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
