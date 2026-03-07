"""Tests for the EIT pipeline configuration module."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from eit_pipeline.config import (
    ASRConfig,
    AudioFileConfig,
    AudioIOConfig,
    DiarizationConfig,
    LeakageCheckConfig,
    OutputConfig,
    PipelineConfig,
    ResponseWindowConfig,
    SilenceDetectionConfig,
    StimulusAlignmentConfig,
    TARGET_SENTENCES,
    VADRefinementConfig,
)


class TestTargetSentences:
    def test_sentence_count(self):
        assert len(TARGET_SENTENCES) == 30

    def test_first_sentence(self):
        assert TARGET_SENTENCES[0] == "Quiero cortarme el pelo"

    def test_last_sentence(self):
        assert TARGET_SENTENCES[29] == "Hay mucha gente que no toma nada para el desayuno"

    def test_all_strings(self):
        for s in TARGET_SENTENCES:
            assert isinstance(s, str)
            assert len(s) > 0


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.audio_io.sample_rate == 16_000
        assert config.asr.backend == "mlx_whisper"
        assert config.diarization.enabled is False
        assert config.leakage_check.enabled is True
        assert len(config.audio_files) == 4

    def test_to_dict(self):
        config = PipelineConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "audio_io" in d
        assert "stimulus_alignment" in d

    def test_to_yaml_roundtrip(self):
        config = PipelineConfig()
        config.experiment_name = "test_roundtrip"

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            tmp_path = f.name

        try:
            config.to_yaml(tmp_path)
            loaded = PipelineConfig.from_yaml(tmp_path)
            assert loaded.experiment_name == "test_roundtrip"
            assert loaded.audio_io.sample_rate == 16_000
            assert loaded.asr.backend == "mlx_whisper"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_to_json(self):
        config = PipelineConfig()
        text = config.to_json()
        data = json.loads(text)
        assert data["audio_io"]["sample_rate"] == 16_000


class TestStimulusAlignmentConfig:
    def test_defaults(self):
        config = StimulusAlignmentConfig()
        assert config.method == "cross_correlation"
        assert config.compare_all_methods is False
        assert config.similarity_threshold == 0.5

    def test_methods(self):
        for method in ["cross_correlation", "mfcc_cosine", "fingerprint"]:
            config = StimulusAlignmentConfig(method=method)
            assert config.method == method


class TestASRConfig:
    def test_defaults(self):
        config = ASRConfig()
        assert config.backend == "mlx_whisper"
        assert config.model_size == "small"
        assert config.language == "es"
        assert config.multi_pass is True


class TestAudioFileConfig:
    def test_creation(self):
        af = AudioFileConfig(
            participant_id="038010",
            eit_version="2A",
            filename="038010_EIT-2A.mp3",
            sheet_name="38010-2A",
            skip_seconds=160.0,
        )
        assert af.participant_id == "038010"
        assert af.skip_seconds == 160.0
