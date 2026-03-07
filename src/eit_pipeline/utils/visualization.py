"""Visualization and debugging utilities for the EIT pipeline.

Provides tools for:
  - Timeline plots showing stimulus events, speech segments, response windows
  - Segment audio playback
  - Alignment comparison visualization
  - Interactive inspection in notebooks
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..audio_io import AudioData
from ..response_window import ResponseSegment
from ..silence_detection import SpeechSegment
from ..stimulus_alignment import AlignmentResult, StimulusEvent

logger = logging.getLogger("eit_pipeline.visualization")


def plot_timeline(
    recording: AudioData,
    stimulus_events: Optional[List[StimulusEvent]] = None,
    speech_segments: Optional[List[SpeechSegment]] = None,
    responses: Optional[List[ResponseSegment]] = None,
    title: str = "EIT Recording Timeline",
    figsize: tuple = (20, 8),
    save_path: Optional[str] = None,
) -> Any:
    """Plot a timeline showing all pipeline stages.

    Parameters
    ----------
    recording : AudioData
        Full recording audio.
    stimulus_events : list of StimulusEvent, optional
        Detected stimulus timestamps.
    speech_segments : list of SpeechSegment, optional
        VAD-detected speech regions.
    responses : list of ResponseSegment, optional
        Response windows.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    duration = recording.duration_s
    time_axis = np.linspace(0, duration, len(recording.waveform))

    # 1. Waveform
    ax = axes[0]
    ax.plot(time_axis, recording.waveform, linewidth=0.3, color="steelblue")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    # 2. Stimulus events
    ax = axes[1]
    ax.set_ylabel("Stimulus")
    ax.set_ylim(0, 1.5)
    if stimulus_events:
        for evt in stimulus_events:
            ax.axvspan(
                evt.stimulus_start_s, evt.stimulus_end_s,
                alpha=0.6, color="coral", label="" if evt != stimulus_events[0] else "Stimulus",
            )
            ax.text(
                (evt.stimulus_start_s + evt.stimulus_end_s) / 2, 1.1,
                str(evt.sentence_id), ha="center", va="bottom", fontsize=7,
            )
    ax.legend(loc="upper right")

    # 3. Speech segments
    ax = axes[2]
    ax.set_ylabel("Speech")
    ax.set_ylim(0, 1.5)
    if speech_segments:
        for seg in speech_segments:
            ax.axvspan(
                seg.start_s, seg.end_s,
                alpha=0.5, color="mediumseagreen",
                label="" if seg != speech_segments[0] else "Speech",
            )
    ax.legend(loc="upper right")

    # 4. Response windows
    ax = axes[3]
    ax.set_ylabel("Response")
    ax.set_ylim(0, 1.5)
    if responses:
        for resp in responses:
            if resp.response_end_s > 0:
                ax.axvspan(
                    resp.response_start_s, resp.response_end_s,
                    alpha=0.5, color="royalblue",
                    label="" if resp != responses[0] else "Response",
                )
                ax.text(
                    (resp.response_start_s + resp.response_end_s) / 2, 1.1,
                    str(resp.sentence_id), ha="center", va="bottom", fontsize=7,
                )
    ax.set_xlabel("Time (seconds)")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved timeline plot: %s", save_path)

    return fig


def plot_alignment_comparison(
    comparison_results: Dict[str, AlignmentResult],
    recording_duration_s: float,
    title: str = "Stimulus Alignment Method Comparison",
    figsize: tuple = (20, 6),
    save_path: Optional[str] = None,
) -> Any:
    """Plot a comparison of alignment methods side by side.

    Parameters
    ----------
    comparison_results : dict
        Method name -> AlignmentResult.
    recording_duration_s : float
        Total recording duration in seconds.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n_methods = len(comparison_results)
    fig, axes = plt.subplots(n_methods, 1, figsize=figsize, sharex=True)
    if n_methods == 1:
        axes = [axes]

    colors = {"cross_correlation": "coral", "mfcc_cosine": "royalblue", "fingerprint": "mediumseagreen"}

    for ax, (method, result) in zip(axes, comparison_results.items()):
        ax.set_ylabel(method)
        ax.set_ylim(0, 1.5)
        ax.set_xlim(0, recording_duration_s)

        color = colors.get(method, "gray")
        for evt in result.events:
            ax.axvspan(
                evt.stimulus_start_s, evt.stimulus_end_s,
                alpha=0.6, color=color,
            )
            ax.text(
                (evt.stimulus_start_s + evt.stimulus_end_s) / 2, 1.1,
                str(evt.sentence_id), ha="center", va="bottom", fontsize=7,
            )

        ax.set_title(f"{method}: {len(result.events)} detections")

    axes[-1].set_xlabel("Time (seconds)")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved comparison plot: %s", save_path)

    return fig


def listen_to_segments(
    responses: List[ResponseSegment],
    recording: AudioData,
    sentence_ids: Optional[List[int]] = None,
) -> None:
    """Play response segments sequentially in a notebook.

    Parameters
    ----------
    responses : list of ResponseSegment
        Response windows.
    recording : AudioData
        Full recording audio.
    sentence_ids : list of int, optional
        Specific sentence IDs to play. If None, plays all.
    """
    try:
        from IPython.display import Audio, display
    except ImportError:
        logger.error("IPython not available — cannot play audio")
        return

    for resp in responses:
        if sentence_ids and resp.sentence_id not in sentence_ids:
            continue

        if resp.response_end_s == 0.0:
            print(f"Sentence {resp.sentence_id}: [no response]")
            continue

        start_sample = int(resp.response_start_s * recording.sample_rate)
        end_sample = int(resp.response_end_s * recording.sample_rate)
        segment = recording.waveform[start_sample:end_sample]

        print(
            f"Sentence {resp.sentence_id}: "
            f"{resp.response_start_s:.1f}s - {resp.response_end_s:.1f}s "
            f"(duration: {resp.response_end_s - resp.response_start_s:.2f}s)"
        )
        display(Audio(data=segment, rate=recording.sample_rate))


def inspect_segment(
    sentence_id: int,
    responses: List[ResponseSegment],
    recording: AudioData,
    speech_segments: Optional[List[SpeechSegment]] = None,
    stimulus_events: Optional[List[StimulusEvent]] = None,
    figsize: tuple = (14, 4),
) -> Any:
    """Detailed inspection of a single response segment.

    Shows a zoomed waveform with VAD and stimulus overlay for a specific
    sentence.

    Parameters
    ----------
    sentence_id : int
        Sentence to inspect.
    responses : list of ResponseSegment
        Response windows.
    recording : AudioData
        Full recording audio.
    speech_segments : list of SpeechSegment, optional
        VAD segments for overlay.
    stimulus_events : list of StimulusEvent, optional
        Stimulus events for overlay.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    resp = next((r for r in responses if r.sentence_id == sentence_id), None)
    if resp is None:
        logger.warning("Sentence %d not found in responses", sentence_id)
        return None

    # Window around the segment
    margin = 3.0  # seconds
    view_start = max(0, resp.stimulus_start_s - margin)
    view_end = min(recording.duration_s, resp.response_end_s + margin) if resp.response_end_s > 0 else resp.stimulus_end_s + 10.0

    start_sample = int(view_start * recording.sample_rate)
    end_sample = int(view_end * recording.sample_rate)
    segment = recording.waveform[start_sample:end_sample]
    time_axis = np.linspace(view_start, view_end, len(segment))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, segment, linewidth=0.5, color="steelblue", alpha=0.7)

    # Stimulus overlay
    ax.axvspan(
        resp.stimulus_start_s, resp.stimulus_end_s,
        alpha=0.3, color="coral", label="Stimulus",
    )

    # Response overlay
    if resp.response_end_s > 0:
        ax.axvspan(
            resp.response_start_s, resp.response_end_s,
            alpha=0.3, color="royalblue", label="Response",
        )

    # Speech segments overlay
    if speech_segments:
        for seg in speech_segments:
            if seg.end_s >= view_start and seg.start_s <= view_end:
                ax.axvspan(
                    max(seg.start_s, view_start), min(seg.end_s, view_end),
                    alpha=0.15, color="mediumseagreen",
                )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Sentence {sentence_id} Detail")
    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig
