"""Self-evaluation module for transcription quality.

Since we have no ground truth for the 4 target participants,
evaluation is multi-pronged:
  - Self-consistency (multi-temperature agreement)
  - Stimulus similarity distribution
  - Confidence score analysis
  - Style comparison with reference transcriptions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import TARGET_SENTENCES, EvaluationConfig
from .experiment_logger import ParticipantResult, TranscriptionResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Evaluation results for one participant."""
    participant_id: str
    # Confidence analysis
    confidence_scores: List[float] = field(default_factory=list)
    avg_confidence: float = 0.0
    std_confidence: float = 0.0
    low_confidence_items: List[int] = field(default_factory=list)  # sentence numbers
    # Stimulus similarity
    stimulus_similarities: List[float] = field(default_factory=list)
    avg_stimulus_similarity: float = 0.0
    high_similarity_items: List[int] = field(default_factory=list)  # possibly transcribed stimulus
    # Self-consistency  
    consistency_scores: List[float] = field(default_factory=list)
    avg_consistency: float = 0.0
    # Style analysis  
    disfluency_count: int = 0
    no_response_count: int = 0
    avg_response_length: float = 0.0
    # Overall assessment
    quality_score: float = 0.0  # 0-1 composite score
    issues: List[str] = field(default_factory=list)


def evaluate_participant(
    result: ParticipantResult,
    config: EvaluationConfig,
    stimuli: Optional[List[str]] = None,
) -> EvaluationReport:
    """Run all evaluation checks on a participant's transcriptions.

    Parameters
    ----------
    result : ParticipantResult
        Transcriptions to evaluate.
    config : EvaluationConfig
        Evaluation configuration.
    stimuli : list of str, optional
        Target sentences (defaults to TARGET_SENTENCES).

    Returns
    -------
    EvaluationReport
    """
    stimuli = stimuli or TARGET_SENTENCES
    report = EvaluationReport(participant_id=result.participant_id)

    # ------------------------------------------------------------------
    # 1. Confidence analysis
    # ------------------------------------------------------------------
    if config.report_confidence_scores:
        _analyse_confidence(result, report)

    # ------------------------------------------------------------------
    # 2. Stimulus similarity
    # ------------------------------------------------------------------
    if config.compute_stimulus_similarity:
        _analyse_stimulus_similarity(result, stimuli, report)

    # ------------------------------------------------------------------
    # 3. Style analysis
    # ------------------------------------------------------------------
    _analyse_style(result, report)

    # ------------------------------------------------------------------
    # 4. Quality score
    # ------------------------------------------------------------------
    _compute_quality_score(report)

    return report


def evaluate_all(
    results: List[ParticipantResult],
    config: EvaluationConfig,
) -> List[EvaluationReport]:
    """Evaluate all participants."""
    reports = []
    for r in results:
        report = evaluate_participant(r, config)
        reports.append(report)
        logger.info(
            "Evaluation for %s: quality=%.2f, avg_conf=%.3f, flagged=%d",
            r.participant_id, report.quality_score, report.avg_confidence,
            len(report.low_confidence_items),
        )
    return reports


def compute_self_consistency(
    transcriptions_multi: List[List[str]],
) -> List[float]:
    """Compute per-item consistency across multiple transcription passes.

    Parameters
    ----------
    transcriptions_multi : list of list of str
        Each inner list is one pass's 30 transcriptions.

    Returns
    -------
    list of float
        Consistency score per sentence (0-1, higher = more consistent).
    """
    if len(transcriptions_multi) < 2:
        return [1.0] * (len(transcriptions_multi[0]) if transcriptions_multi else 0)

    n_items = len(transcriptions_multi[0])
    scores = []

    for i in range(n_items):
        items = [passes[i] for passes in transcriptions_multi if i < len(passes)]
        if len(items) < 2:
            scores.append(1.0)
            continue

        # Pairwise word-level agreement
        total_sim = 0.0
        count = 0
        for j in range(len(items)):
            for k in range(j + 1, len(items)):
                total_sim += _word_similarity(items[j], items[k])
                count += 1

        scores.append(total_sim / max(count, 1))

    return scores


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _analyse_confidence(result: ParticipantResult, report: EvaluationReport) -> None:
    """Analyse confidence scores across transcriptions."""
    scores = [t.avg_log_prob for t in result.transcriptions]
    report.confidence_scores = scores

    if scores:
        report.avg_confidence = float(np.mean(scores))
        report.std_confidence = float(np.std(scores))
        # Flag items with very low confidence
        threshold = report.avg_confidence - 2 * report.std_confidence
        for t in result.transcriptions:
            if t.avg_log_prob < max(threshold, -2.0):
                report.low_confidence_items.append(t.sentence_number)


def _analyse_stimulus_similarity(
    result: ParticipantResult,
    stimuli: List[str],
    report: EvaluationReport,
) -> None:
    """Check if transcriptions are too similar to stimuli."""
    similarities = []

    for t in result.transcriptions:
        idx = t.sentence_number - 1
        if 0 <= idx < len(stimuli):
            sim = _word_similarity(t.transcription, stimuli[idx])
        else:
            sim = 0.0
        similarities.append(sim)

        # Flag very high similarity (might have transcribed stimulus instead)
        if sim > 0.95:
            report.high_similarity_items.append(t.sentence_number)
            report.issues.append(
                f"Sentence {t.sentence_number}: similarity to stimulus = {sim:.2f} "
                f"(may have transcribed stimulus instead of response)"
            )

    report.stimulus_similarities = similarities
    report.avg_stimulus_similarity = float(np.mean(similarities)) if similarities else 0.0


def _analyse_style(result: ParticipantResult, report: EvaluationReport) -> None:
    """Analyse disfluency patterns and response characteristics."""
    disf_count = 0
    no_resp_count = 0
    lengths = []

    for t in result.transcriptions:
        text = t.transcription

        # Count disfluencies
        disf_markers = ['...', '[pause]', 'xxx', '[gibberish]', '[no response]', '-']
        for marker in disf_markers:
            disf_count += text.count(marker)

        if text == "[no response]":
            no_resp_count += 1
        else:
            lengths.append(len(text.split()))

    report.disfluency_count = disf_count
    report.no_response_count = no_resp_count
    report.avg_response_length = float(np.mean(lengths)) if lengths else 0.0


def _compute_quality_score(report: EvaluationReport) -> None:
    """Compute a composite quality score (0-1).

    Heuristic based on:
    - Confidence level
    - Number of problematic items
    - Reasonable stimulus similarity distribution
    """
    score = 1.0

    # Penalise low confidence
    if report.avg_confidence < -1.0:
        score -= 0.2
    elif report.avg_confidence < -0.7:
        score -= 0.1

    # Penalise too many flagged items
    n = len(report.confidence_scores)
    if n > 0:
        flag_ratio = len(report.low_confidence_items) / n
        score -= flag_ratio * 0.3

    # Penalise if all similarities are very high (transcribed stimuli?)
    if report.avg_stimulus_similarity > 0.9:
        score -= 0.4
        report.issues.append(
            "Very high average similarity to stimuli — check segmentation"
        )
    elif report.avg_stimulus_similarity > 0.8:
        score -= 0.2

    # Good sign: some variation in similarity (expected for learner responses)
    if report.stimulus_similarities:
        sim_std = float(np.std(report.stimulus_similarities))
        if sim_std > 0.1:
            score += 0.05  # bonus for expected variation

    report.quality_score = max(0.0, min(1.0, score))


def _word_similarity(text1: str, text2: str) -> float:
    """Word-level Jaccard similarity."""
    import re
    words1 = set(re.sub(r'[^\w\s]', '', text1.lower()).split())
    words2 = set(re.sub(r'[^\w\s]', '', text2.lower()).split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0
