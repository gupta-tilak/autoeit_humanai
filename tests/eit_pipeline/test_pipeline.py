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
        # response is anchored on the VAD segment start (with tiny padding back)
        assert responses[0].response_start_s <= 3.0
        assert responses[0].response_start_s >  2.0  # never in stimulus territory
        assert responses[1].sentence_id == 2
        assert responses[2].sentence_id == 3

    def test_no_speech_for_stimulus(self):
        """If no VAD segment falls inside the safe zone, result is no-response.

        This is the hybrid strategy: stimulus timestamps define where to look,
        but a real voiced anchor is required to define the response.
        """
        stimuli = self._make_stimulus_events()
        speech = [
            SpeechSegment(3.0, 5.0),   # only in the gap after stimulus 1
        ]

        responses = build_response_windows(stimuli, speech)
        assert len(responses) == 3
        # Stimulus 1 produces a response (VAD found)
        assert responses[0].response_end_s > 0.0
        # Stimuli 2 and 3 have no VAD in their zones → no-response
        assert responses[1].response_end_s == 0.0
        assert responses[2].response_end_s == 0.0
        assert responses[1].speech_segments_used == 0
        assert responses[2].speech_segments_used == 0

    def test_empty_stimuli(self):
        responses = build_response_windows([], [SpeechSegment(1.0, 2.0)])
        assert len(responses) == 0

    def test_response_doesnt_overlap_next_stimulus(self):
        stimuli = [
            StimulusEvent(1, 0.0, 2.0, 0.9, "test"),
            StimulusEvent(2, 5.0, 7.0, 0.9, "test"),
        ]
        # Speech segment crosses into the next stimulus zone; should be capped.
        speech = [
            SpeechSegment(2.5, 6.0),  # starts in zone 1, but very long
        ]

        responses = build_response_windows(stimuli, speech)
        # Response 1 must end before stimulus 2 starts
        assert responses[0].response_end_s <= stimuli[1].stimulus_start_s

    def test_duration_constraints(self):
        """Windows shorter than min_response_duration_s are rejected.

        Two stimuli placed very close together produce a tiny inter-stimulus
        gap; after subtracting the pre/post gaps the remaining window is shorter
        than the minimum and should be marked as no-response.
        """
        # S1 ends at 2.0 s; S2 starts at 2.7 s.
        # response_start = 2.0 + 0.4 = 2.4
        # response_end   = 2.7 - 0.4 = 2.3  →  duration = -0.1 s  (invalid)
        stimuli = [
            StimulusEvent(1, 0.0, 2.0, 0.9, "test"),
            StimulusEvent(2, 2.7, 4.5, 0.9, "test"),
        ]
        speech = [SpeechSegment(2.1, 2.5)]  # small speech burst in tiny gap

        config = ResponseWindowConfig(min_response_duration_s=0.7)
        responses = build_response_windows(stimuli, speech, config)
        # Stimulus 1 window is too short → marked as no response
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
