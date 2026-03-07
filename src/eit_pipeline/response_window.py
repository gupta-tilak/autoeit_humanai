"""Stage 4 — Response Window Construction.

Constructs one response window per stimulus event using a *hybrid* strategy
that combines stimulus timestamps with VAD speech segments.

Algorithm for stimulus i
─────────────────────────
1. Compute a safe search zone from stimulus timestamps:

       zone_start = stimulus_end_i   + post_stimulus_gap_s
       zone_end   = stimulus_start_(i+1) - pre_stimulus_gap_s

   The gaps push the zone clear of both the trailing edge of the current
   stimulus and the leading edge of the next one, tolerating small timestamp
   errors from the alignment stage.

2. Within the zone, find the first VAD speech segment whose start falls
   inside [zone_start, zone_end].

3. Merge consecutive speech segments that belong to the same response
   (pauses ≤ merge_gap_s, e.g. the learner hesitates mid-sentence).

4. Cap merged end at zone_end so no audio from the next stimulus leaks in.

5. Apply hard duration constraints [min_response_duration_s,
   max_response_duration_s].

6. Apply small padding (padding_before_s / padding_after_s) for a cleaner
   ASR context.

If no VAD segment is found inside the zone the entry is marked as no-response
(response_start_s == response_end_s == 0.0).  Stage 5 (vad_refinement) then
refines the exact voiced boundaries within the already-clean segment.

Key guarantee
─────────────
ASR *never* sees stimulus audio:
  - The zone_start gap keeps stimulus audio out even if timestamps slip
  - Only real voiced-speech anchors (VAD) define the response start
  - The zone_end cap prevents bleed into the next stimulus
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
    """Construct response windows using a hybrid stimulus-timestamp + VAD strategy.

    For each stimulus event:
    1. Compute a safe zone from stimulus timestamps
    2. Find the first VAD speech segment inside the zone
    3. Merge adjacent segments (learner may pause mid-sentence)
    4. Cap at zone boundary and apply duration constraints

    Parameters
    ----------
    stimulus_events : list of StimulusEvent
        Detected stimulus timestamps, sorted by time.
    speech_segments : list of SpeechSegment
        VAD speech segments — used as the actual response anchors.
    config : ResponseWindowConfig, optional
        Configuration parameters.

    Returns
    -------
    list of ResponseSegment
        One entry per stimulus event.  No-response entries have
        response_start_s == response_end_s == 0.0.
    """
    if config is None:
        config = ResponseWindowConfig()

    if not stimulus_events:
        logger.warning("No stimulus events provided — cannot build response windows")
        return []

    stimulus_events = sorted(stimulus_events, key=lambda e: e.stimulus_start_s)
    speech_segments = sorted(speech_segments, key=lambda s: s.start_s)

    responses = []

    for i, stim in enumerate(stimulus_events):
        # ── 1. Safe search zone from stimulus timestamps ─────────────────
        zone_start = stim.stimulus_end_s + config.post_stimulus_gap_s
        if i + 1 < len(stimulus_events):
            zone_end = stimulus_events[i + 1].stimulus_start_s - config.pre_stimulus_gap_s
        else:
            zone_end = stim.stimulus_end_s + config.max_response_duration_s

        zone_start = max(0.0, zone_start)
        zone_end   = max(0.0, zone_end)

        # Degenerate zone — stimuli too close together
        if zone_end <= zone_start:
            logger.debug(
                "Degenerate zone for stimulus %d (zone_start=%.2f >= zone_end=%.2f) — no response",
                stim.sentence_id, zone_start, zone_end,
            )
            responses.append(_empty(stim))
            continue

        # ── 2. Find first VAD speech segment inside the zone ─────────────
        # A segment is a candidate if its start lies inside [zone_start, zone_end).
        candidates = [
            s for s in speech_segments
            if s.start_s >= zone_start and s.start_s < zone_end
        ]

        if not candidates:
            logger.debug(
                "No VAD speech found in zone for stimulus %d (%.2f–%.2fs) — no response",
                stim.sentence_id, zone_start, zone_end,
            )
            responses.append(_empty(stim))
            continue

        # ── 3. Merge adjacent segments (same response, learner pauses) ───
        first_seg = candidates[0]
        response_start = first_seg.start_s
        response_end   = first_seg.end_s

        for seg in candidates[1:]:
            gap = seg.start_s - response_end
            if gap <= config.merge_gap_s:
                response_end = max(response_end, seg.end_s)
            else:
                break  # gap too large — next segment is a different event

        # ── 4. Cap at zone boundary ──────────────────────────────────────
        response_end = min(response_end, zone_end)

        # ── 5. Apply small padding ───────────────────────────────────────
        response_start = max(zone_start, response_start - config.padding_before_s)
        response_end   = min(zone_end,   response_end   + config.padding_after_s)

        # ── 6. Duration constraints ──────────────────────────────────────
        duration = response_end - response_start

        if duration < config.min_response_duration_s:
            logger.debug(
                "Response for stimulus %d too short (%.2fs < %.2fs) — no response",
                stim.sentence_id, duration, config.min_response_duration_s,
            )
            responses.append(_empty(stim))
            continue

        if duration > config.max_response_duration_s:
            response_end = response_start + config.max_response_duration_s
            logger.debug(
                "Capped response for stimulus %d to %.1fs",
                stim.sentence_id, config.max_response_duration_s,
            )

        logger.debug(
            "Stimulus %d: zone %.2f–%.2fs | response %.2f–%.2fs (%.2fs, %d VAD segs)",
            stim.sentence_id, zone_start, zone_end,
            response_start, response_end,
            response_end - response_start, len(candidates),
        )

        responses.append(ResponseSegment(
            sentence_id=stim.sentence_id,
            stimulus_start_s=stim.stimulus_start_s,
            stimulus_end_s=stim.stimulus_end_s,
            response_start_s=response_start,
            response_end_s=response_end,
            confidence=stim.confidence,
            speech_segments_used=len(candidates),
        ))

    responses = _validate_no_overlap(responses, stimulus_events)

    logger.info(
        "Built %d response windows (%d with speech, %d no-response)",
        len(responses),
        sum(1 for r in responses if r.response_end_s > 0),
        sum(1 for r in responses if r.response_end_s == 0),
    )
    return responses


def _empty(stim: StimulusEvent) -> ResponseSegment:
    """Return a no-response placeholder linked to a stimulus event."""
    return ResponseSegment(
        sentence_id=stim.sentence_id,
        stimulus_start_s=stim.stimulus_start_s,
        stimulus_end_s=stim.stimulus_end_s,
        response_start_s=0.0,
        response_end_s=0.0,
        confidence=0.0,
        speech_segments_used=0,
    )


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
