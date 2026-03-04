"""Main pipeline orchestrator.

Connects all modules into a single end-to-end pipeline:
  Audio → Preprocessing → Segmentation → ASR → Post-Processing → Output

Fully configurable, logged, and reproducible.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .asr import ASRBackend, create_asr_backend
from .asr.base import ASRResult
from .config import (
    AudioFileConfig,
    PipelineConfig,
    TARGET_SENTENCES,
)
from .evaluation import evaluate_all, evaluate_participant, compute_self_consistency
from .experiment_logger import (
    ExperimentLogger,
    ExperimentRun,
    ParticipantResult,
    TranscriptionResult,
)
from .output import write_results_to_excel, write_detailed_results
from .postprocessing import postprocess_transcription
from .preprocessing import preprocess_audio, compute_rms
from .segmentation import segment_audio, AudioSegment

logger = logging.getLogger(__name__)


class AutoEITPipeline:
    """End-to-end pipeline for AutoEIT transcription.

    Usage::

        config = PipelineConfig(...)
        pipeline = AutoEITPipeline(config, base_dir="/path/to/project")
        pipeline.run()
    """

    def __init__(
        self,
        config: PipelineConfig,
        base_dir: Optional[str] = None,
    ):
        self.config = config
        self.base_dir = base_dir or os.getcwd()

        # Resolve paths
        self.resolved_config = config.resolve_paths(self.base_dir)

        # Setup logging
        self._setup_logging()

        # Experiment logger
        self.exp_logger = ExperimentLogger(self.resolved_config.experiments_dir)

        # ASR backend (created lazily)
        self._asr: Optional[ASRBackend] = None

        # Results storage
        self.participant_results: List[ParticipantResult] = []
        self.evaluation_reports = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        participant_ids: Optional[List[str]] = None,
        name: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> ExperimentRun:
        """Execute the full pipeline.

        Parameters
        ----------
        participant_ids : list of str, optional
            Process only these participants. If None, process all.
        name : str
            Experiment name.
        description : str
            Experiment description.
        tags : list of str, optional
            Tags for this run.

        Returns
        -------
        ExperimentRun
            Complete experiment run with all results.
        """
        name = name or self.config.experiment_name
        description = description or self.config.experiment_description
        tags = tags or self.config.tags

        # Start experiment tracking
        run = self.exp_logger.start_run(
            config=self.config,
            name=name,
            description=description,
            tags=tags,
        )

        try:
            # 1. Initialize ASR backend
            logger.info("=" * 60)
            logger.info("AUTOEIT PIPELINE — %s", name)
            logger.info("=" * 60)
            self._init_asr()

            # 2. Filter audio files
            audio_files = self.config.audio_files
            if participant_ids:
                audio_files = [
                    af for af in audio_files
                    if af.participant_id in participant_ids
                ]

            # 3. Process each participant
            for af in audio_files:
                try:
                    result = self._process_participant(af)
                    self.participant_results.append(result)
                    self.exp_logger.log_participant(run, result)
                except Exception as e:
                    logger.error("Failed to process %s: %s", af.participant_id, e, exc_info=True)
                    run.errors.append(f"{af.participant_id}: {str(e)}")

            # 4. Evaluation
            logger.info("Running evaluation...")
            self.evaluation_reports = evaluate_all(
                self.participant_results, self.config.evaluation,
            )

            # 5. Generate output
            self._generate_output()

            # 6. Finalize experiment
            self.exp_logger.finish_run(run)
            run_dir = self.exp_logger.save(run)
            logger.info("Experiment saved to: %s", run_dir)

        except Exception as e:
            logger.error("Pipeline failed: %s", e, exc_info=True)
            run.errors.append(str(e))
            self.exp_logger.finish_run(run)
            self.exp_logger.save(run)
            raise

        return run

    def run_single_file(
        self,
        audio_file_config: AudioFileConfig,
    ) -> ParticipantResult:
        """Process a single audio file (useful for testing/debugging)."""
        self._init_asr()
        return self._process_participant(audio_file_config)

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _init_asr(self) -> None:
        """Initialize the ASR backend if not already done."""
        if self._asr is not None and self._asr.is_loaded():
            return

        logger.info("Initializing ASR backend: %s", self.config.asr.backend)
        self._asr = create_asr_backend(self.config.asr)
        self._asr.load_model()
        logger.info("ASR backend ready: %s", self._asr.backend_name)
        logger.info("Model info: %s", self._asr.model_info)

    def _run_full_file_asr(self, wav_path: str) -> List[Dict[str, Any]]:
        """Run ASR on the full preprocessed audio to get word-level timestamps.

        Returns a list of dicts: [{"word": str, "start": float, "end": float}, ...]
        """
        logger.info("Running full-file ASR on: %s", wav_path)
        t0 = time.time()

        asr_result = self._asr.transcribe(
            wav_path,
            language=self.config.asr.language,
            initial_prompt=self.config.asr.initial_prompt,
            temperature=0.0,
            beam_size=self.config.asr.beam_size,
        )

        # Extract word-level timeline from ASR segments
        word_timeline: List[Dict[str, Any]] = []
        for seg in asr_result.segments:
            for w in seg.words:
                word_timeline.append({
                    "word": w.get("word", w) if isinstance(w, dict) else str(w),
                    "start": w.get("start", 0.0) if isinstance(w, dict) else 0.0,
                    "end": w.get("end", 0.0) if isinstance(w, dict) else 0.0,
                })

        elapsed = time.time() - t0
        logger.info(
            "Full-file ASR done: %d words, %.1fs (transcript: %d chars)",
            len(word_timeline), elapsed, len(asr_result.text),
        )
        return word_timeline

    def _process_participant(self, af: AudioFileConfig) -> ParticipantResult:
        """Full pipeline for one participant."""
        logger.info("-" * 50)
        logger.info("Processing participant %s (%s)", af.participant_id, af.filename)
        logger.info("-" * 50)

        t0 = time.time()

        # Initialize result
        result = ParticipantResult(
            participant_id=af.participant_id,
            eit_version=af.eit_version,
            sheet_name=af.sheet_name,
            filename=af.filename,
            skip_seconds=af.skip_seconds,
        )

        # 1. Preprocess audio
        audio_path = os.path.join(self.resolved_config.audio_dir, af.filename)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, sr, wav_path = preprocess_audio(
            audio_path, af, self.config.preprocessing,
            cache_dir=self.resolved_config.cache_dir,
        )
        logger.info("Preprocessed: %.1fs audio", len(audio) / sr)

        # 2. Full-file ASR (provides word-level timestamps for segmentation)
        word_timeline = self._run_full_file_asr(wav_path)
        logger.info("Full-file ASR: %d words with timestamps", len(word_timeline))

        # 3. Segment audio (with word timeline for text-matching strategy)
        segments = segment_audio(audio, sr, self.config.segmentation, word_timeline=word_timeline)
        result.segments_found = len(segments)
        result.segmentation_method = "phase1_vad_tone_textmatch"
        logger.info("Segmented: %d segments found", len(segments))

        # 4. Transcribe each segment
        # Always use per-segment ASR for primary transcription.
        # Full-file ASR pre_transcription is used as fallback when
        # per-segment ASR produces [no response] but pre_transcription has text.
        transcriptions = []
        for seg in segments:
            trans = self._transcribe_segment(seg, af)

            # Fallback: if per-segment ASR gave no_response but full-file
            # ASR detected text in this window, override with that text.
            if (
                trans.raw_transcription.strip() == ""
                and seg.pre_transcription
                and seg.notes != "no_response"
            ):
                logger.debug(
                    "Sentence %d: per-segment ASR empty, using full-file pre-trans",
                    seg.sentence_number,
                )
                trans.raw_transcription = seg.pre_transcription
                trans.transcription = ""  # will be set by post-processing

            transcriptions.append(trans)

        # 5. Post-process
        for trans in transcriptions:
            idx = trans.sentence_number - 1
            stimulus = TARGET_SENTENCES[idx] if 0 <= idx < len(TARGET_SENTENCES) else ""

            processed_text, flagged, flag_reason = postprocess_transcription(
                raw_text=trans.raw_transcription,
                stimulus=stimulus,
                config=self.config.postprocessing,
                avg_log_prob=trans.avg_log_prob,
                no_speech_prob=trans.no_speech_prob,
                segment_energy=seg.energy_rms if seg.sentence_number == trans.sentence_number else 0.0,
            )
            trans.transcription = processed_text
            trans.flagged = flagged
            trans.flag_reason = flag_reason

        result.transcriptions = transcriptions
        result.total_processing_time_s = time.time() - t0

        # Compute summary metrics
        if transcriptions:
            probs = [t.avg_log_prob for t in transcriptions]
            result.avg_confidence = float(np.mean(probs))
            result.low_confidence_count = sum(1 for t in transcriptions if t.flagged)
            result.no_response_count = sum(
                1 for t in transcriptions if t.transcription == "[no response]"
            )

        logger.info(
            "Participant %s done: %d transcriptions, avg_conf=%.3f, "
            "flagged=%d, no_response=%d, time=%.1fs",
            af.participant_id,
            len(transcriptions),
            result.avg_confidence,
            result.low_confidence_count,
            result.no_response_count,
            result.total_processing_time_s,
        )

        return result

    def _transcribe_segment(
        self,
        segment: AudioSegment,
        af: AudioFileConfig,
    ) -> TranscriptionResult:
        """Transcribe a single audio segment with multi-pass strategy."""
        idx = segment.sentence_number - 1
        stimulus = TARGET_SENTENCES[idx] if 0 <= idx < len(TARGET_SENTENCES) else ""

        t0 = time.time()

        # Handle no-response segments
        if segment.audio is None or len(segment.audio) == 0 or segment.notes == "no_response":
            return TranscriptionResult(
                sentence_number=segment.sentence_number,
                stimulus=stimulus,
                transcription="[no response]",
                raw_transcription="",
                avg_log_prob=0.0,
                no_speech_prob=1.0,
                processing_time_s=0.0,
                pass_number=0,
                segment_start_s=segment.start_s,
                segment_end_s=segment.end_s,
            )

        # Check energy level
        rms = compute_rms(segment.audio)
        if rms < self.config.postprocessing.no_response_energy_threshold:
            return TranscriptionResult(
                sentence_number=segment.sentence_number,
                stimulus=stimulus,
                transcription="[no response]",
                raw_transcription="",
                avg_log_prob=0.0,
                no_speech_prob=1.0,
                processing_time_s=time.time() - t0,
                pass_number=0,
                segment_start_s=segment.start_s,
                segment_end_s=segment.end_s,
            )

        # First pass
        asr_result = self._asr.transcribe_array(
            segment.audio,
            sample_rate=self.config.preprocessing.sample_rate,
            language=self.config.asr.language,
            initial_prompt=self.config.asr.initial_prompt,
            temperature=self.config.asr.temperature,
            beam_size=self.config.asr.beam_size,
        )

        pass_number = 1
        best_result = asr_result

        # Multi-pass: re-transcribe if confidence is low
        if (
            self.config.asr.multi_pass
            and asr_result.avg_log_prob < self.config.asr.low_confidence_threshold
        ):
            logger.debug(
                "Low confidence (%.3f) for sentence %d — running second pass",
                asr_result.avg_log_prob, segment.sentence_number,
            )
            second_result = self._asr.transcribe_array(
                segment.audio,
                sample_rate=self.config.preprocessing.sample_rate,
                language=self.config.asr.language,
                initial_prompt=self.config.asr.initial_prompt,
                temperature=self.config.asr.second_pass_temperature,
                beam_size=self.config.asr.beam_size,
            )
            # Keep the result with higher confidence
            if second_result.avg_log_prob > asr_result.avg_log_prob:
                best_result = second_result
                pass_number = 2

        elapsed = time.time() - t0

        return TranscriptionResult(
            sentence_number=segment.sentence_number,
            stimulus=stimulus,
            transcription="",  # filled in post-processing
            raw_transcription=best_result.text,
            avg_log_prob=best_result.avg_log_prob,
            no_speech_prob=best_result.no_speech_prob,
            processing_time_s=elapsed,
            pass_number=pass_number,
            segment_start_s=segment.start_s,
            segment_end_s=segment.end_s,
        )

    def _generate_output(self) -> None:
        """Generate all output files."""
        if not self.participant_results:
            logger.warning("No results to output")
            return

        output_dir = Path(self.resolved_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Populate the template Excel
        template_path = self.resolved_config.template_excel
        if os.path.exists(template_path):
            output_excel = str(output_dir / "AutoEIT_Transcriptions.xlsx")
            write_results_to_excel(
                self.participant_results, template_path, output_excel,
            )
        else:
            logger.warning("Template Excel not found: %s", template_path)

        # 2. Detailed results Excel
        detailed_path = str(output_dir / "AutoEIT_Detailed_Results.xlsx")
        write_detailed_results(self.participant_results, detailed_path)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        """Configure logging for the pipeline."""
        log_dir = Path(self.resolved_config.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Root logger
        root = logging.getLogger("autoeit")
        root.setLevel(logging.DEBUG)

        # Console handler
        if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            ))
            root.addHandler(ch)

        # File handler
        log_path = log_dir / "pipeline.log"
        if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            fh = logging.FileHandler(str(log_path), mode="a")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            ))
            root.addHandler(fh)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_pipeline(
    config: Optional[PipelineConfig] = None,
    base_dir: Optional[str] = None,
    name: str = "default",
    description: str = "",
    tags: Optional[List[str]] = None,
    config_path: Optional[str] = None,
) -> ExperimentRun:
    """Convenience function to run the full pipeline.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration. If not provided, loads from *config_path*
        or uses defaults.
    base_dir : str, optional
        Project root directory.
    name : str
        Experiment name.
    config_path : str, optional
        Path to a YAML config file.

    Returns
    -------
    ExperimentRun
    """
    if config is None:
        if config_path:
            config = PipelineConfig.from_yaml(config_path)
        else:
            config = PipelineConfig()

    config.experiment_name = name
    config.experiment_description = description
    config.tags = tags or []

    pipeline = AutoEITPipeline(config, base_dir=base_dir)
    return pipeline.run(name=name, description=description, tags=tags)
