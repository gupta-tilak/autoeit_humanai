"""Tests for stimulus alignment methods."""

import numpy as np
import pytest

from eit_pipeline.stimulus_alignment import (
    AlignmentResult,
    StimulusEvent,
    _post_process_events,
    align_stimuli,
    load_stimulus_audio,
)
from eit_pipeline.audio_io import AudioData
from eit_pipeline.config import StimulusAlignmentConfig


class TestPostProcessEvents:
    def test_sort_by_time(self):
        events = [
            StimulusEvent(2, 10.0, 12.0, 0.8, "test"),
            StimulusEvent(1, 0.0, 2.0, 0.9, "test"),
            StimulusEvent(3, 20.0, 22.0, 0.7, "test"),
        ]
        config = StimulusAlignmentConfig()
        result = _post_process_events(events, config)
        assert result[0].sentence_id == 1
        assert result[1].sentence_id == 2
        assert result[2].sentence_id == 3

    def test_remove_duplicate_sentence_ids(self):
        events = [
            StimulusEvent(1, 0.0, 2.0, 0.9, "test"),
            StimulusEvent(1, 5.0, 7.0, 0.5, "test"),  # duplicate, lower confidence
        ]
        config = StimulusAlignmentConfig()
        result = _post_process_events(events, config)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_enforce_spacing(self):
        events = [
            StimulusEvent(1, 0.0, 2.0, 0.9, "test"),
            StimulusEvent(2, 2.5, 4.5, 0.8, "test"),  # too close (< 5s spacing)
        ]
        config = StimulusAlignmentConfig(min_stimulus_spacing_s=5.0)
        result = _post_process_events(events, config)
        # Higher confidence one should survive
        assert len(result) == 1
        assert result[0].sentence_id == 1

    def test_empty_events(self):
        config = StimulusAlignmentConfig()
        result = _post_process_events([], config)
        assert result == []


class TestAlignStimuli:
    def test_invalid_method_raises(self):
        recording = AudioData(
            waveform=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            duration_s=1.0,
            source_path="test.wav",
            original_sr=16000,
        )
        config = StimulusAlignmentConfig(method="invalid_method")
        with pytest.raises(ValueError, match="Unknown alignment method"):
            align_stimuli(recording, {}, config)

    def test_no_stimulus_audio(self):
        recording = AudioData(
            waveform=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            duration_s=1.0,
            source_path="test.wav",
            original_sr=16000,
        )
        config = StimulusAlignmentConfig()
        result = align_stimuli(recording, {}, config)
        assert len(result.events) == 0


class TestLoadStimulusAudio:
    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            load_stimulus_audio("/nonexistent/path")
