"""Tests for response window construction and leakage check."""

import pytest

from eit_pipeline.response_window import ResponseSegment, build_response_windows
from eit_pipeline.silence_detection import SpeechSegment
from eit_pipeline.stimulus_alignment import StimulusEvent
from eit_pipeline.config import ResponseWindowConfig


class TestResponseWindowBuilder:
    def _make_stimulus_events(self):
        """Create test stimulus events."""
        return [
            StimulusEvent(1, 0.0, 2.0, 0.9, "test"),
            StimulusEvent(2, 10.0, 12.0, 0.9, "test"),
            StimulusEvent(3, 20.0, 22.0, 0.9, "test"),
        ]

    def test_basic_response_assignment(self):
        stimuli = self._make_stimulus_events()
        speech = [
            SpeechSegment(3.0, 5.0),   # response to stimulus 1
            SpeechSegment(13.0, 15.0),  # response to stimulus 2
            SpeechSegment(23.0, 25.0),  # response to stimulus 3
        ]

        responses = build_response_windows(stimuli, speech)
        assert len(responses) == 3
        assert responses[0].sentence_id == 1
        assert responses[0].response_start_s > 0
        assert responses[1].sentence_id == 2
        assert responses[2].sentence_id == 3

    def test_no_speech_for_stimulus(self):
        stimuli = self._make_stimulus_events()
        speech = [
            SpeechSegment(3.0, 5.0),   # response to stimulus 1 only
        ]

        responses = build_response_windows(stimuli, speech)
        assert len(responses) == 3
        # Stimulus 2 and 3 should have no response
        assert responses[1].response_end_s == 0.0
        assert responses[2].response_end_s == 0.0

    def test_empty_stimuli(self):
        responses = build_response_windows([], [SpeechSegment(1.0, 2.0)])
        assert len(responses) == 0

    def test_response_doesnt_overlap_next_stimulus(self):
        stimuli = [
            StimulusEvent(1, 0.0, 2.0, 0.9, "test"),
            StimulusEvent(2, 5.0, 7.0, 0.9, "test"),
        ]
        speech = [
            SpeechSegment(2.5, 6.0),  # overlaps with stimulus 2
        ]

        responses = build_response_windows(stimuli, speech)
        # Response for stimulus 1 should be trimmed before stimulus 2
        assert responses[0].response_end_s <= stimuli[1].stimulus_start_s

    def test_duration_constraints(self):
        stimuli = [StimulusEvent(1, 0.0, 2.0, 0.9, "test")]
        speech = [SpeechSegment(2.1, 2.15)]  # very short

        config = ResponseWindowConfig(min_response_duration_s=0.5)
        responses = build_response_windows(stimuli, speech, config)
        # Should be marked as no response due to short duration
        assert responses[0].response_end_s == 0.0


class TestLeakageCheck:
    def test_leakage_detection(self):
        from eit_pipeline.leakage_check import check_leakage, LeakageFlag
        from eit_pipeline.asr_transcription import TranscriptionResult
        from eit_pipeline.config import LeakageCheckConfig

        transcriptions = [
            TranscriptionResult(
                sentence_id=1,
                response_start_s=3.0,
                response_end_s=5.0,
                transcript="Quiero cortarme el pelo",  # exact match!
                raw_transcript="Quiero cortarme el pelo",
            ),
        ]

        config = LeakageCheckConfig(
            enabled=True,
            similarity_threshold=0.85,
            similarity_method="rapidfuzz",
        )

        flags = check_leakage(transcriptions, config)
        assert len(flags) == 1
        assert flags[0].flagged is True
        assert flags[0].similarity_score > 0.85

    def test_no_leakage(self):
        from eit_pipeline.leakage_check import check_leakage
        from eit_pipeline.asr_transcription import TranscriptionResult
        from eit_pipeline.config import LeakageCheckConfig

        transcriptions = [
            TranscriptionResult(
                sentence_id=1,
                response_start_s=3.0,
                response_end_s=5.0,
                transcript="cortarme pelo",  # partial, different
                raw_transcript="cortarme pelo",
            ),
        ]

        config = LeakageCheckConfig(
            enabled=True,
            similarity_threshold=0.85,
        )

        flags = check_leakage(transcriptions, config)
        assert len(flags) == 1
        assert flags[0].flagged is False
