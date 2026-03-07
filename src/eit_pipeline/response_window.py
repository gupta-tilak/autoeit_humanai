"""Stage 4 — Response Window Construction.

Associates detected speech segments with stimulus events to identify
learner response windows.

Rule: A response segment must start after stimulus_end and before next
stimulus_start.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from .config import ResponseWindowConfig
from .silence_detection import SpeechSegment
from .stimulus_alignment import StimulusEvent

logger = logging.getLogger("eit_pipeline.response_window")


@dataclass
class ResponseSegment:
    """A learner response segment linked to its stimulus."""
    sentence_id: int
    stimulus_start_s: float
    stimulus_end_s: float
    response_start_s: float
    response_end_s: float
    confidence: float = 1.0
    speech_segments_used: int = 0


def build_response_windows(
    stimulus_events: List[StimulusEvent],
    speech_segments: List[SpeechSegment],
    config: Optional[ResponseWindowConfig] = None,
) -> List[ResponseSegment]:
    """Construct response windows from stimulus events and speech segments.

    For each stimulus event, find the first speech segment(s) that occur
    after the stimulus ends and before the next stimulus begins.

    Parameters
    ----------
    stimulus_events : list of StimulusEvent
        Detected stimulus timestamps, sorted by time.
    speech_segments : list of SpeechSegment
        Detected speech segments, sorted by time.
    config : ResponseWindowConfig, optional
        Configuration parameters.

    Returns
    -------
    list of ResponseSegment
        Response windows, one per stimulus event.
    """
    if config is None:
        config = ResponseWindowConfig()

    if not stimulus_events:
        logger.warning("No stimulus events provided — cannot build response windows")
        return []

    # Ensure sorted
    stimulus_events = sorted(stimulus_events, key=lambda e: e.stimulus_start_s)
    speech_segments = sorted(speech_segments, key=lambda s: s.start_s)

    responses = []

    for i, stim in enumerate(stimulus_events):
        # Define the window in which a response is allowable
        window_start = stim.stimulus_end_s
        if i + 1 < len(stimulus_events):
            window_end = stimulus_events[i + 1].stimulus_start_s
        else:
            # Last stimulus: allow response up to max duration after stimulus
            window_end = stim.stimulus_end_s + config.max_response_duration_s

        # Find speech segments within this window
        candidate_segments = [
            seg for seg in speech_segments
            if seg.start_s >= window_start - 0.1  # small tolerance
            and seg.start_s < window_end
        ]

        if not candidate_segments:
            # No speech detected in response window
            logger.debug(
                "No response found for stimulus %d (window: %.1f-%.1fs)",
                stim.sentence_id, window_start, window_end,
            )
            # Still create a response entry marking no response
            responses.append(ResponseSegment(
                sentence_id=stim.sentence_id,
                stimulus_start_s=stim.stimulus_start_s,
                stimulus_end_s=stim.stimulus_end_s,
                response_start_s=0.0,
                response_end_s=0.0,
                confidence=0.0,
                speech_segments_used=0,
            ))
            continue

        # Use the first speech segment as the primary response
        first_seg = candidate_segments[0]

        # Determine response end: could span multiple speech segments
        # if they are within the same response window
        response_start = first_seg.start_s
        response_end = first_seg.end_s

        # Merge consecutive candidate segments that belong to same response
        for seg in candidate_segments[1:]:
            if seg.start_s <= response_end + config.padding_after_s + 0.5:
                response_end = max(response_end, seg.end_s)
            else:
                break

        # Cap response end at window boundary
        response_end = min(response_end, window_end)

        # Apply padding
        response_start = max(0, response_start - config.padding_before_s)
        response_end = response_end + config.padding_after_s

        # Validate duration
        duration = response_end - response_start
        if duration < config.min_response_duration_s:
            logger.debug(
                "Response for stimulus %d too short (%.2fs < %.2fs)",
                stim.sentence_id, duration, config.min_response_duration_s,
            )
            responses.append(ResponseSegment(
                sentence_id=stim.sentence_id,
                stimulus_start_s=stim.stimulus_start_s,
                stimulus_end_s=stim.stimulus_end_s,
                response_start_s=0.0,
                response_end_s=0.0,
                confidence=0.0,
                speech_segments_used=0,
            ))
            continue

        if duration > config.max_response_duration_s:
            response_end = response_start + config.max_response_duration_s
            logger.debug(
                "Capped response for stimulus %d to %.1fs",
                stim.sentence_id, config.max_response_duration_s,
            )

        responses.append(ResponseSegment(
            sentence_id=stim.sentence_id,
            stimulus_start_s=stim.stimulus_start_s,
            stimulus_end_s=stim.stimulus_end_s,
            response_start_s=response_start,
            response_end_s=response_end,
            confidence=stim.confidence,
            speech_segments_used=len(candidate_segments),
        ))

        logger.debug(
            "Response for stimulus %d: %.1f-%.1fs (%d speech segments)",
            stim.sentence_id, response_start, response_end,
            len(candidate_segments),
        )

    # Validate: responses should not overlap with next stimulus
    responses = _validate_no_overlap(responses, stimulus_events)

    logger.info(
        "Built %d response windows (%d with speech)",
        len(responses),
        sum(1 for r in responses if r.response_end_s > 0),
    )
    return responses


def _validate_no_overlap(
    responses: List[ResponseSegment],
    stimulus_events: List[StimulusEvent],
) -> List[ResponseSegment]:
    """Ensure no response overlaps with the next stimulus."""
    stim_by_id = {e.sentence_id: e for e in stimulus_events}

    for i, resp in enumerate(responses):
        if resp.response_end_s == 0.0:
            continue

        # Find next stimulus
        next_stim = None
        for j in range(i + 1, len(responses)):
            next_sid = responses[j].sentence_id
            if next_sid in stim_by_id:
                next_stim = stim_by_id[next_sid]
                break

        if next_stim and resp.response_end_s > next_stim.stimulus_start_s:
            logger.warning(
                "Response %d overlaps stimulus %d — trimming (%.1f -> %.1f)",
                resp.sentence_id, next_stim.sentence_id,
                resp.response_end_s, next_stim.stimulus_start_s,
            )
            resp.response_end_s = next_stim.stimulus_start_s

    return responses
