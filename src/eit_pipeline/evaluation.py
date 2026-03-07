"""Evaluation utilities for the EIT pipeline.

Provides metrics for assessing pipeline quality: stimulus matching accuracy,
response window quality, transcription confidence analysis, and method
comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .asr_transcription import TranscriptionResult
from .leakage_check import LeakageFlag
from .output import ParticipantOutput
from .response_window import ResponseSegment
from .stimulus_alignment import AlignmentResult

logger = logging.getLogger("eit_pipeline.evaluation")


@dataclass
class PipelineMetrics:
    """Aggregated pipeline quality metrics."""
    # Stimulus alignment
    stimuli_detected: int = 0
    stimuli_expected: int = 30
    alignment_confidence_mean: float = 0.0
    alignment_confidence_std: float = 0.0
    # Response windows
    responses_with_speech: int = 0
    responses_no_speech: int = 0
    avg_response_duration_s: float = 0.0
    # Transcription
    avg_log_prob: float = 0.0
    avg_no_speech_prob: float = 0.0
    # Leakage
    leakage_flagged: int = 0
    # Overall
    total_processing_time_s: float = 0.0
    method_used: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def compute_metrics(
    alignment: AlignmentResult,
    responses: List[ResponseSegment],
    transcriptions: List[TranscriptionResult],
    leakage_flags: List[LeakageFlag],
    expected_stimuli: int = 30,
) -> PipelineMetrics:
    """Compute comprehensive pipeline metrics.

    Parameters
    ----------
    alignment : AlignmentResult
        Stimulus alignment result.
    responses : list of ResponseSegment
        Response windows.
    transcriptions : list of TranscriptionResult
        ASR transcription results.
    leakage_flags : list of LeakageFlag
        Leakage check results.
    expected_stimuli : int
        Expected number of stimuli.

    Returns
    -------
    PipelineMetrics
    """
    metrics = PipelineMetrics(
        stimuli_expected=expected_stimuli,
        method_used=alignment.method,
    )

    # Stimulus alignment metrics
    metrics.stimuli_detected = len(alignment.events)
    if alignment.events:
        confs = [e.confidence for e in alignment.events]
        metrics.alignment_confidence_mean = float(np.mean(confs))
        metrics.alignment_confidence_std = float(np.std(confs))

    # Response window metrics
    responses_with = [r for r in responses if r.response_end_s > 0]
    metrics.responses_with_speech = len(responses_with)
    metrics.responses_no_speech = len(responses) - len(responses_with)

    if responses_with:
        durations = [r.response_end_s - r.response_start_s for r in responses_with]
        metrics.avg_response_duration_s = float(np.mean(durations))

    # Transcription metrics
    text_transcriptions = [
        t for t in transcriptions if t.transcript != "[no response]"
    ]
    if text_transcriptions:
        metrics.avg_log_prob = float(np.mean([t.avg_log_prob for t in text_transcriptions]))
        metrics.avg_no_speech_prob = float(np.mean([t.no_speech_prob for t in text_transcriptions]))

    # Leakage metrics
    metrics.leakage_flagged = sum(1 for f in leakage_flags if f.flagged)

    # Processing time
    metrics.total_processing_time_s = sum(t.processing_time_s for t in transcriptions)

    return metrics


def compare_alignment_methods(
    comparison_results: Dict[str, AlignmentResult],
    expected_stimuli: int = 30,
) -> Dict[str, Dict]:
    """Compare stimulus alignment methods.

    Parameters
    ----------
    comparison_results : dict
        Method name -> AlignmentResult from compare_all_methods().
    expected_stimuli : int
        Expected number of stimuli.

    Returns
    -------
    dict
        Method name -> metrics dict.
    """
    comparison = {}

    for method, result in comparison_results.items():
        events = result.events
        detected = len(events)
        confs = [e.confidence for e in events] if events else [0.0]

        # Check temporal ordering
        ordered = True
        for i in range(1, len(events)):
            if events[i].stimulus_start_s <= events[i - 1].stimulus_start_s:
                ordered = False
                break

        # Check coverage: do we find all 30 sentence IDs?
        found_ids = set(e.sentence_id for e in events)
        missing_ids = set(range(1, expected_stimuli + 1)) - found_ids

        comparison[method] = {
            "stimuli_detected": detected,
            "stimuli_expected": expected_stimuli,
            "detection_rate": detected / expected_stimuli if expected_stimuli > 0 else 0,
            "confidence_mean": float(np.mean(confs)),
            "confidence_std": float(np.std(confs)),
            "confidence_min": float(np.min(confs)),
            "confidence_max": float(np.max(confs)),
            "temporally_ordered": ordered,
            "missing_sentence_ids": sorted(missing_ids),
            "error": result.metadata.get("error"),
        }

    return comparison


def format_metrics_report(metrics: PipelineMetrics) -> str:
    """Format metrics as a human-readable report."""
    lines = [
        "=" * 60,
        "EIT PIPELINE METRICS REPORT",
        "=" * 60,
        f"Method: {metrics.method_used}",
        "",
        "--- Stimulus Alignment ---",
        f"  Detected:   {metrics.stimuli_detected} / {metrics.stimuli_expected}",
        f"  Confidence: {metrics.alignment_confidence_mean:.3f} ± {metrics.alignment_confidence_std:.3f}",
        "",
        "--- Response Windows ---",
        f"  With speech:  {metrics.responses_with_speech}",
        f"  No speech:    {metrics.responses_no_speech}",
        f"  Avg duration: {metrics.avg_response_duration_s:.2f}s",
        "",
        "--- Transcription ---",
        f"  Avg log prob:     {metrics.avg_log_prob:.3f}",
        f"  Avg no-speech:    {metrics.avg_no_speech_prob:.3f}",
        "",
        "--- Quality ---",
        f"  Leakage flagged:  {metrics.leakage_flagged}",
        f"  Processing time:  {metrics.total_processing_time_s:.1f}s",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_comparison_report(comparison: Dict[str, Dict]) -> str:
    """Format method comparison as a human-readable report."""
    lines = [
        "=" * 70,
        "STIMULUS ALIGNMENT METHOD COMPARISON",
        "=" * 70,
    ]

    for method, stats in comparison.items():
        lines.append(f"\n--- {method.upper()} ---")
        if stats.get("error"):
            lines.append(f"  ERROR: {stats['error']}")
            continue
        lines.append(f"  Detection rate: {stats['detection_rate']:.1%} ({stats['stimuli_detected']}/{stats['stimuli_expected']})")
        lines.append(f"  Confidence:     {stats['confidence_mean']:.3f} ± {stats['confidence_std']:.3f}")
        lines.append(f"  Range:          [{stats['confidence_min']:.3f}, {stats['confidence_max']:.3f}]")
        lines.append(f"  Ordered:        {stats['temporally_ordered']}")
        if stats["missing_sentence_ids"]:
            lines.append(f"  Missing IDs:    {stats['missing_sentence_ids']}")

    lines.append("=" * 70)
    return "\n".join(lines)
