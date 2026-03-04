"""Experiment logging system for AutoEIT.

Tracks every pipeline run with:
  - Full configuration snapshot
  - ASR backend & model details
  - Per-sentence transcription results + confidence scores
  - Timing & performance metrics
  - Self-evaluation results
  - Free-form notes

Logs are persisted as structured JSON so that all experimental iterations
can be reviewed, compared, and exported later.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    """Result for a single sentence transcription."""
    sentence_number: int                # 1-30
    stimulus: str                       # target sentence
    transcription: str                  # ASR output after post-processing
    raw_transcription: str = ""         # ASR output before post-processing
    avg_log_prob: float = 0.0           # Whisper confidence
    no_speech_prob: float = 0.0
    processing_time_s: float = 0.0
    pass_number: int = 1                # which ASR pass produced this
    segment_start_s: float = 0.0        # start time in original audio
    segment_end_s: float = 0.0          # end time in original audio
    flagged: bool = False               # low confidence / needs review
    flag_reason: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParticipantResult:
    """All results for one participant audio file."""
    participant_id: str
    eit_version: str
    sheet_name: str
    filename: str
    skip_seconds: float
    total_processing_time_s: float = 0.0
    segmentation_method: str = ""
    segments_found: int = 0
    transcriptions: List[TranscriptionResult] = field(default_factory=list)
    # Evaluation metrics
    avg_confidence: float = 0.0
    low_confidence_count: int = 0
    no_response_count: int = 0
    self_consistency_score: float = 0.0
    stimulus_similarity_scores: List[float] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ExperimentRun:
    """A complete experiment run."""
    # Identity
    run_id: str = ""
    experiment_name: str = ""
    experiment_description: str = ""
    tags: List[str] = field(default_factory=list)

    # Timestamps
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0

    # Environment
    python_version: str = ""
    platform_info: str = ""
    machine: str = ""
    cpu: str = ""

    # Configuration snapshot (stored as dict)
    config: Dict[str, Any] = field(default_factory=dict)

    # ASR backend details
    asr_backend: str = ""
    asr_model: str = ""
    asr_model_path: str = ""

    # Results
    participants: List[ParticipantResult] = field(default_factory=list)

    # Overall metrics
    total_sentences: int = 0
    total_flagged: int = 0
    total_no_response: int = 0
    overall_avg_confidence: float = 0.0
    total_processing_time_s: float = 0.0

    # Notes
    notes: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Experiment Logger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """Manages experiment tracking across pipeline runs.

    Usage::

        logger = ExperimentLogger(experiments_dir="experiments")
        run = logger.start_run(config=my_config, name="baseline_v1")
        # ... pipeline work ...
        logger.log_participant(run, participant_result)
        logger.finish_run(run)
        logger.save(run)
    """

    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._runs_index_path = self.experiments_dir / "runs_index.json"
        self._runs_index = self._load_runs_index()

    # -------------------------------------------------------------------
    # Run lifecycle
    # -------------------------------------------------------------------

    def start_run(
        self,
        config: Any,
        name: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> ExperimentRun:
        """Create and register a new experiment run."""
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%d_%H%M%S") + f"_{name}" if name else now.strftime("%Y%m%d_%H%M%S")

        config_dict = config.to_dict() if hasattr(config, "to_dict") else {}

        run = ExperimentRun(
            run_id=run_id,
            experiment_name=name or config_dict.get("experiment_name", "unnamed"),
            experiment_description=description or config_dict.get("experiment_description", ""),
            tags=tags or config_dict.get("tags", []),
            started_at=now.isoformat(),
            python_version=platform.python_version(),
            platform_info=f"{platform.system()} {platform.release()}",
            machine=platform.machine(),
            cpu=self._get_cpu_info(),
            config=config_dict,
            asr_backend=config_dict.get("asr", {}).get("backend", ""),
            asr_model=config_dict.get("asr", {}).get("model_size", ""),
            asr_model_path=config_dict.get("asr", {}).get("model_path", "") or "",
        )
        logger.info("Started experiment run: %s", run_id)
        return run

    def log_participant(self, run: ExperimentRun, result: ParticipantResult) -> None:
        """Add a participant's results to the run."""
        run.participants.append(result)
        logger.info(
            "Logged participant %s — %d transcriptions, avg confidence %.3f",
            result.participant_id,
            len(result.transcriptions),
            result.avg_confidence,
        )

    def finish_run(self, run: ExperimentRun) -> None:
        """Finalise timing and aggregate metrics."""
        now = datetime.now(timezone.utc)
        run.finished_at = now.isoformat()

        if run.started_at:
            start = datetime.fromisoformat(run.started_at)
            run.duration_s = (now - start).total_seconds()

        # Aggregate
        total_sentences = 0
        total_flagged = 0
        total_no_resp = 0
        confidence_sum = 0.0
        confidence_count = 0
        total_time = 0.0

        for p in run.participants:
            total_sentences += len(p.transcriptions)
            total_flagged += sum(1 for t in p.transcriptions if t.flagged)
            total_no_resp += p.no_response_count
            for t in p.transcriptions:
                confidence_sum += t.avg_log_prob
                confidence_count += 1
            total_time += p.total_processing_time_s

        run.total_sentences = total_sentences
        run.total_flagged = total_flagged
        run.total_no_response = total_no_resp
        run.overall_avg_confidence = confidence_sum / max(confidence_count, 1)
        run.total_processing_time_s = total_time

        logger.info(
            "Finished run %s — %d sentences, %d flagged, %.1fs total",
            run.run_id, total_sentences, total_flagged, run.duration_s,
        )

    def save(self, run: ExperimentRun) -> Path:
        """Persist the run to disk and update the index."""
        run_dir = self.experiments_dir / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Full run data
        run_path = run_dir / "run.json"
        run_path.write_text(run.to_json(), encoding="utf-8")

        # Config snapshot
        config_path = run_dir / "config.json"
        config_path.write_text(
            json.dumps(run.config, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # Per-participant CSV-style summaries
        for p in run.participants:
            p_path = run_dir / f"{p.participant_id}_{p.eit_version}_transcriptions.json"
            p_path.write_text(
                json.dumps(p.to_dict(), indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        # Summary file (human-readable)
        summary = self._generate_summary(run)
        summary_path = run_dir / "summary.txt"
        summary_path.write_text(summary, encoding="utf-8")

        # Update index
        self._update_index(run)

        logger.info("Saved run %s to %s", run.run_id, run_dir)
        return run_dir

    # -------------------------------------------------------------------
    # Comparison & retrieval
    # -------------------------------------------------------------------

    def list_runs(self) -> List[Dict[str, Any]]:
        """Return the index of all logged runs."""
        return list(self._runs_index.get("runs", []))

    def load_run(self, run_id: str) -> ExperimentRun:
        """Load a previous run from disk."""
        run_path = self.experiments_dir / run_id / "run.json"
        if not run_path.exists():
            raise FileNotFoundError(f"Run {run_id} not found at {run_path}")
        data = json.loads(run_path.read_text(encoding="utf-8"))
        return self._dict_to_run(data)

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Generate a comparison table across multiple runs."""
        comparison = {"runs": []}
        for rid in run_ids:
            run = self.load_run(rid)
            comparison["runs"].append({
                "run_id": run.run_id,
                "name": run.experiment_name,
                "backend": run.asr_backend,
                "model": run.asr_model,
                "total_sentences": run.total_sentences,
                "total_flagged": run.total_flagged,
                "total_no_response": run.total_no_response,
                "avg_confidence": round(run.overall_avg_confidence, 4),
                "duration_s": round(run.duration_s, 1),
                "tags": run.tags,
            })
        return comparison

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _generate_summary(self, run: ExperimentRun) -> str:
        lines = [
            f"AutoEIT Experiment Run Summary",
            f"{'=' * 50}",
            f"Run ID:       {run.run_id}",
            f"Name:         {run.experiment_name}",
            f"Description:  {run.experiment_description}",
            f"Tags:         {', '.join(run.tags)}",
            f"Started:      {run.started_at}",
            f"Finished:     {run.finished_at}",
            f"Duration:     {run.duration_s:.1f}s",
            f"",
            f"ASR Backend:  {run.asr_backend}",
            f"ASR Model:    {run.asr_model}",
            f"Model Path:   {run.asr_model_path}",
            f"",
            f"Environment:",
            f"  Python:     {run.python_version}",
            f"  Platform:   {run.platform_info}",
            f"  Machine:    {run.machine}",
            f"  CPU:        {run.cpu}",
            f"",
            f"Results:",
            f"  Total sentences:      {run.total_sentences}",
            f"  Total flagged:        {run.total_flagged}",
            f"  Total no response:    {run.total_no_response}",
            f"  Avg confidence:       {run.overall_avg_confidence:.4f}",
            f"  Processing time:      {run.total_processing_time_s:.1f}s",
            f"",
        ]

        for p in run.participants:
            lines.append(f"Participant {p.participant_id} ({p.eit_version}):")
            lines.append(f"  Segments found:  {p.segments_found}")
            lines.append(f"  Avg confidence:  {p.avg_confidence:.4f}")
            lines.append(f"  Low confidence:  {p.low_confidence_count}")
            lines.append(f"  No response:     {p.no_response_count}")
            lines.append(f"  Processing time: {p.total_processing_time_s:.1f}s")
            lines.append("")
            for t in p.transcriptions:
                flag = " [FLAGGED]" if t.flagged else ""
                lines.append(f"  [{t.sentence_number:2d}] {t.transcription}{flag}")
            lines.append("")

        if run.errors:
            lines.append("Errors:")
            for e in run.errors:
                lines.append(f"  - {e}")
            lines.append("")

        if run.notes:
            lines.append(f"Notes: {run.notes}")

        return "\n".join(lines)

    def _load_runs_index(self) -> Dict[str, Any]:
        if self._runs_index_path.exists():
            return json.loads(self._runs_index_path.read_text(encoding="utf-8"))
        return {"runs": []}

    def _update_index(self, run: ExperimentRun) -> None:
        entry = {
            "run_id": run.run_id,
            "name": run.experiment_name,
            "backend": run.asr_backend,
            "model": run.asr_model,
            "started_at": run.started_at,
            "duration_s": round(run.duration_s, 1),
            "total_sentences": run.total_sentences,
            "total_flagged": run.total_flagged,
            "avg_confidence": round(run.overall_avg_confidence, 4),
            "tags": run.tags,
        }
        # Replace if same run_id exists
        self._runs_index["runs"] = [
            r for r in self._runs_index["runs"] if r["run_id"] != run.run_id
        ]
        self._runs_index["runs"].append(entry)
        self._runs_index_path.write_text(
            json.dumps(self._runs_index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _dict_to_run(d: Dict[str, Any]) -> ExperimentRun:
        """Reconstruct an ExperimentRun from a dict."""
        participants_data = d.pop("participants", [])
        participants = []
        for pd_item in participants_data:
            transcriptions_data = pd_item.pop("transcriptions", [])
            transcriptions = [TranscriptionResult(**t) for t in transcriptions_data]
            pd_item["transcriptions"] = transcriptions
            participants.append(ParticipantResult(**pd_item))
        d["participants"] = participants
        return ExperimentRun(**d)

    @staticmethod
    def _get_cpu_info() -> str:
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return platform.processor() or "unknown"
