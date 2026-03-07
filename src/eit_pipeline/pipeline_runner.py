"""Pipeline Runner — Main orchestrator for the EIT pipeline.

Coordinates all stages in the correct order:
  1. Audio Loading
  2. Stimulus Audio Matching
  3. Silence Detection
  4. Response Window Construction
  5. VAD Boundary Refinement
  6. (Optional) Diarization Filter
  7. ASR Transcription
  8. Stimulus Leakage Check
  9. Final Output
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audio_io import AudioData, load_audio
from .asr_transcription import TranscriptionResult, transcribe_responses
from .config import AudioFileConfig, PipelineConfig, TARGET_SENTENCES
from .diarization_filter import filter_by_diarization
from .evaluation import (
    PipelineMetrics,
    compare_alignment_methods,
    compute_metrics,
    format_comparison_report,
    format_metrics_report,
)
from .leakage_check import LeakageFlag, check_leakage
from .output import ParticipantOutput, assemble_results, export_all
from .response_window import ResponseSegment, build_response_windows
from .silence_detection import SpeechSegment, detect_speech
from .stimulus_alignment import (
    AlignmentResult,
    StimulusEvent,
    align_stimuli,
    compare_all_methods,
    load_stimulus_audio,
)
from .vad_refinement import refine_boundaries

logger = logging.getLogger("eit_pipeline.pipeline_runner")


class EITPipeline:
    """Main pipeline orchestrator.

    Runs all stages in sequence for each participant recording.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._stimulus_audio = None

    def load_stimulus_references(self) -> None:
        """Load stimulus reference audio files.

        Must be called before processing recordings.
        The stimulus audio directory should contain files named
        stimulus_01.wav, stimulus_02.wav, etc.
        """
        stim_dir = self.config.stimulus_alignment.stimulus_audio_dir
        if not Path(stim_dir).exists():
            raise FileNotFoundError(
                f"Stimulus audio directory not found: {stim_dir}\n"
                f"Extract stimulus audio first using the stimulus_extractor module."
            )

        self._stimulus_audio = load_stimulus_audio(
            stim_dir, self.config.audio_io.sample_rate,
        )

        if not self._stimulus_audio:
            raise RuntimeError(f"No stimulus audio files found in {stim_dir}")

        logger.info("Loaded %d stimulus reference files", len(self._stimulus_audio))

    def process_recording(
        self,
        audio_file: AudioFileConfig,
    ) -> Dict[str, Any]:
        """Process a single participant recording through all pipeline stages.

        Parameters
        ----------
        audio_file : AudioFileConfig
            Metadata for the recording to process.

        Returns
        -------
        dict
            Complete results including all intermediate outputs for inspection.
        """
        if self._stimulus_audio is None:
            raise RuntimeError("Stimulus references not loaded. Call load_stimulus_references() first.")

        participant_id = audio_file.participant_id
        logger.info("=" * 60)
        logger.info(
            "Processing: %s (version %s)", participant_id, audio_file.eit_version,
        )
        logger.info("=" * 60)

        pipeline_start = time.time()

        audio_path = str(
            Path(self.config.audio_dir) / audio_file.filename
        )

        # ── Stage 1: Audio Loading ─────────────────────────────
        logger.info("Stage 1: Loading audio...")
        recording = load_audio(
            audio_path,
            config=self.config.audio_io,
            skip_seconds=audio_file.skip_seconds,
        )
        logger.info("  Audio: %.1fs @ %d Hz", recording.duration_s, recording.sample_rate)

        # ── Stage 2: Stimulus Alignment ────────────────────────
        logger.info("Stage 2: Stimulus alignment...")
        comparison_results = None

        if self.config.stimulus_alignment.compare_all_methods:
            logger.info("  Running all 3 methods for comparison...")
            comparison_results = compare_all_methods(
                recording, self._stimulus_audio, self.config.stimulus_alignment,
            )
            # Use the configured primary method for downstream processing
            alignment = comparison_results.get(
                self.config.stimulus_alignment.method,
                list(comparison_results.values())[0],
            )
        else:
            alignment = align_stimuli(
                recording, self._stimulus_audio, self.config.stimulus_alignment,
            )

        logger.info("  Detected %d stimulus events", len(alignment.events))

        # ── Stage 3: Silence Detection ─────────────────────────
        logger.info("Stage 3: Silence detection...")
        speech_segments = detect_speech(recording, self.config.silence_detection)
        logger.info("  Detected %d speech segments", len(speech_segments))

        # ── Stage 4: Response Window Construction ──────────────
        logger.info("Stage 4: Building response windows...")
        responses = build_response_windows(
            alignment.events, speech_segments, self.config.response_window,
        )
        responses_with_speech = sum(1 for r in responses if r.response_end_s > 0)
        logger.info(
            "  Built %d windows (%d with speech)", len(responses), responses_with_speech,
        )

        # ── Stage 5: VAD Boundary Refinement ───────────────────
        logger.info("Stage 5: VAD boundary refinement...")
        responses = refine_boundaries(
            responses, recording, self.config.vad_refinement,
        )

        # ── Stage 6: Optional Diarization Filter ───────────────
        diarization_result = None
        if self.config.diarization.enabled:
            logger.info("Stage 6: Diarization filter...")
            responses, diarization_result = filter_by_diarization(
                responses, recording, self.config.diarization,
            )
        else:
            logger.info("Stage 6: Diarization disabled — skipping")

        # ── Stage 7: ASR Transcription ─────────────────────────
        logger.info("Stage 7: ASR transcription...")
        transcriptions = transcribe_responses(
            responses, recording, self.config.asr,
        )

        # ── Stage 8: Stimulus Leakage Check ────────────────────
        logger.info("Stage 8: Leakage check...")
        leakage_flags = check_leakage(
            transcriptions, self.config.leakage_check,
        )

        # ── Stage 9: Final Output ─────────────────────────────
        logger.info("Stage 9: Assembling output...")
        output = assemble_results(
            participant_id, audio_file.eit_version,
            responses, transcriptions, leakage_flags,
        )

        # Compute metrics
        metrics = compute_metrics(
            alignment, responses, transcriptions, leakage_flags,
        )
        metrics.total_processing_time_s = time.time() - pipeline_start

        # Export
        output_dir = str(
            Path(self.config.output_dir) / participant_id
        )
        export_paths = export_all(output, output_dir, self.config.output)

        # Log summary
        report = format_metrics_report(metrics)
        logger.info("\n%s", report)

        # Build full result for inspection
        result = {
            "participant_id": participant_id,
            "eit_version": audio_file.eit_version,
            "recording": recording,
            "alignment": alignment,
            "comparison_results": comparison_results,
            "speech_segments": speech_segments,
            "responses": responses,
            "transcriptions": transcriptions,
            "leakage_flags": leakage_flags,
            "output": output,
            "metrics": metrics,
            "export_paths": export_paths,
            "diarization_result": diarization_result,
        }

        return result

    def process_all(self) -> List[Dict[str, Any]]:
        """Process all configured recordings.

        Returns
        -------
        list of dict
            Results for each participant.
        """
        results = []

        for audio_file in self.config.audio_files:
            try:
                result = self.process_recording(audio_file)
                results.append(result)
            except Exception as e:
                logger.error(
                    "Failed to process %s: %s",
                    audio_file.participant_id, e, exc_info=True,
                )
                results.append({
                    "participant_id": audio_file.participant_id,
                    "error": str(e),
                })

        # Save summary
        self._save_run_summary(results)

        return results

    def _save_run_summary(self, results: List[Dict]) -> None:
        """Save a summary of the pipeline run."""
        summary_dir = Path(self.config.output_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "experiment_name": self.config.experiment_name,
            "experiment_description": self.config.experiment_description,
            "tags": self.config.tags,
            "config": self.config.to_dict(),
            "participants": [],
        }

        for result in results:
            if "error" in result:
                summary["participants"].append({
                    "participant_id": result["participant_id"],
                    "error": result["error"],
                })
            else:
                metrics = result["metrics"]
                summary["participants"].append({
                    "participant_id": result["participant_id"],
                    "eit_version": result["eit_version"],
                    "stimuli_detected": metrics.stimuli_detected,
                    "responses_with_speech": metrics.responses_with_speech,
                    "avg_log_prob": metrics.avg_log_prob,
                    "leakage_flagged": metrics.leakage_flagged,
                    "processing_time_s": metrics.total_processing_time_s,
                })

        summary_path = summary_dir / "run_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Run summary saved: %s", summary_path)


def run_pipeline(
    config: Optional[PipelineConfig] = None,
    config_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience function to run the full pipeline.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.
    config_path : str, optional
        Path to YAML config file.

    Returns
    -------
    list of dict
        Results for each participant.
    """
    if config is None and config_path:
        config = PipelineConfig.from_yaml(config_path)
    elif config is None:
        config = PipelineConfig()

    pipeline = EITPipeline(config)
    pipeline.load_stimulus_references()
    return pipeline.process_all()
