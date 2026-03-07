"""Stage 2 — Stimulus Audio Matching.

Detects exact timestamps where stimulus sentences occur in a full recording
using reference stimulus audio files.

Implements three methods:
  A) Spectrogram cross-correlation
  B) MFCC cosine similarity sliding window
  C) Audio fingerprinting

All methods can be run independently or compared side-by-side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy import signal as scipy_signal

from .audio_io import AudioData
from .config import StimulusAlignmentConfig

logger = logging.getLogger("eit_pipeline.stimulus_alignment")


@dataclass
class StimulusEvent:
    """A detected stimulus occurrence in the recording."""
    sentence_id: int
    stimulus_start_s: float
    stimulus_end_s: float
    confidence: float
    method: str


@dataclass
class AlignmentResult:
    """Complete result from stimulus alignment."""
    events: List[StimulusEvent]
    method: str
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def align_stimuli(
    recording: AudioData,
    stimulus_audio: Dict[int, Tuple[np.ndarray, int]],
    config: Optional[StimulusAlignmentConfig] = None,
) -> AlignmentResult:
    """Detect stimulus timestamps in a recording.

    Parameters
    ----------
    recording : AudioData
        Full participant recording.
    stimulus_audio : dict
        Mapping of sentence_id -> (waveform, sample_rate) for each stimulus.
    config : StimulusAlignmentConfig, optional
        Configuration parameters.

    Returns
    -------
    AlignmentResult
        Detected stimulus events sorted by time.
    """
    if config is None:
        config = StimulusAlignmentConfig()

    method_map = {
        "cross_correlation": _align_cross_correlation,
        "mfcc_cosine": _align_mfcc_cosine,
        "fingerprint": _align_fingerprint,
    }

    if config.method not in method_map:
        raise ValueError(
            f"Unknown alignment method: {config.method}. "
            f"Available: {list(method_map.keys())}"
        )

    logger.info("Running stimulus alignment with method: %s", config.method)
    result = method_map[config.method](recording, stimulus_audio, config)

    # Post-process: sort by time, remove duplicates, enforce spacing
    result.events = _post_process_events(result.events, config)

    # Ensure all expected stimuli are present (interpolate missing ones)
    n_expected = config.expected_stimuli if hasattr(config, 'expected_stimuli') else 30
    result.events = _ensure_all_stimuli(
        result.events,
        n_expected=n_expected,
        recording_duration_s=recording.duration_s,
    )

    logger.info(
        "Stimulus alignment complete: %d/%d events (%d detected, %d interpolated)",
        len(result.events), n_expected,
        sum(1 for e in result.events if e.method != "interpolated"),
        sum(1 for e in result.events if e.method == "interpolated"),
    )
    return result


def compare_all_methods(
    recording: AudioData,
    stimulus_audio: Dict[int, Tuple[np.ndarray, int]],
    config: Optional[StimulusAlignmentConfig] = None,
) -> Dict[str, AlignmentResult]:
    """Run all three alignment methods and return results for comparison.

    Parameters
    ----------
    recording : AudioData
        Full participant recording.
    stimulus_audio : dict
        Mapping of sentence_id -> (waveform, sample_rate) for each stimulus.
    config : StimulusAlignmentConfig, optional
        Configuration parameters.

    Returns
    -------
    dict
        Method name -> AlignmentResult for each method.
    """
    if config is None:
        config = StimulusAlignmentConfig()

    results = {}
    methods = ["cross_correlation", "mfcc_cosine", "fingerprint"]

    for method in methods:
        logger.info("Running comparison method: %s", method)
        method_config = StimulusAlignmentConfig(**{
            **config.__dict__, "method": method
        })
        try:
            results[method] = align_stimuli(
                recording, stimulus_audio, method_config,
            )
        except Exception as e:
            logger.error("Method %s failed: %s", method, e)
            results[method] = AlignmentResult(events=[], method=method, metadata={"error": str(e)})

    return results


def load_stimulus_audio(
    stimulus_dir: str,
    sample_rate: int = 16_000,
) -> Dict[int, Tuple[np.ndarray, int]]:
    """Load reference stimulus audio files from a directory.

    Expects files named stimulus_01.wav, stimulus_02.wav, etc.

    Parameters
    ----------
    stimulus_dir : str
        Path to directory containing stimulus WAV files.
    sample_rate : int
        Target sample rate for loading.

    Returns
    -------
    dict
        sentence_id -> (waveform, sample_rate)
    """
    stim_dir = Path(stimulus_dir)
    if not stim_dir.exists():
        raise FileNotFoundError(f"Stimulus directory not found: {stimulus_dir}")

    stimuli = {}
    for wav_file in sorted(stim_dir.glob("stimulus_*.wav")):
        # Extract sentence ID from filename
        try:
            sid = int(wav_file.stem.split("_")[1])
        except (IndexError, ValueError):
            logger.warning("Skipping unrecognised file: %s", wav_file.name)
            continue

        waveform, sr = librosa.load(str(wav_file), sr=sample_rate, mono=True)
        stimuli[sid] = (waveform.astype(np.float32), sr)
        logger.debug("Loaded stimulus %d: %.2fs", sid, len(waveform) / sr)

    logger.info("Loaded %d stimulus audio files from %s", len(stimuli), stimulus_dir)
    return stimuli


# ---------------------------------------------------------------------------
# Method A: Spectrogram Cross-Correlation
# ---------------------------------------------------------------------------

def _align_cross_correlation(
    recording: AudioData,
    stimulus_audio: Dict[int, Tuple[np.ndarray, int]],
    config: StimulusAlignmentConfig,
) -> AlignmentResult:
    """Detect stimuli using mel-spectrogram cross-correlation."""
    events = []

    # Compute recording mel spectrogram
    rec_mel = librosa.feature.melspectrogram(
        y=recording.waveform,
        sr=recording.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
    )
    rec_mel_db = librosa.power_to_db(rec_mel, ref=np.max)

    for sid, (stim_wav, stim_sr) in sorted(stimulus_audio.items()):
        # Compute stimulus mel spectrogram
        stim_mel = librosa.feature.melspectrogram(
            y=stim_wav,
            sr=stim_sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        stim_mel_db = librosa.power_to_db(stim_mel, ref=np.max)

        # Flatten mel bands for 1-D cross-correlation
        # Average across mel bands to get a 1-D pattern
        rec_profile = np.mean(rec_mel_db, axis=0)
        stim_profile = np.mean(stim_mel_db, axis=0)

        # Normalise
        rec_norm = (rec_profile - np.mean(rec_profile)) / (np.std(rec_profile) + 1e-8)
        stim_norm = (stim_profile - np.mean(stim_profile)) / (np.std(stim_profile) + 1e-8)

        # Cross-correlate
        correlation = np.correlate(rec_norm, stim_norm, mode="valid")
        correlation = correlation / (len(stim_norm) + 1e-8)

        # Find peaks
        stim_duration_frames = len(stim_profile)
        min_distance_frames = int(
            config.peak_distance_s * recording.sample_rate / config.hop_length
        )

        if len(correlation) == 0:
            continue

        peaks, properties = scipy_signal.find_peaks(
            correlation,
            prominence=config.peak_prominence,
            distance=min_distance_frames,
        )

        for peak_idx in peaks:
            confidence = float(correlation[peak_idx])
            if confidence < config.similarity_threshold:
                continue

            start_frame = peak_idx
            end_frame = peak_idx + stim_duration_frames
            start_s = float(start_frame * config.hop_length / recording.sample_rate)
            end_s = float(end_frame * config.hop_length / recording.sample_rate)

            events.append(StimulusEvent(
                sentence_id=sid,
                stimulus_start_s=start_s,
                stimulus_end_s=end_s,
                confidence=confidence,
                method="cross_correlation",
            ))

    return AlignmentResult(events=events, method="cross_correlation")


# ---------------------------------------------------------------------------
# Method B: MFCC Cosine Similarity Sliding Window
# ---------------------------------------------------------------------------

def _align_mfcc_cosine(
    recording: AudioData,
    stimulus_audio: Dict[int, Tuple[np.ndarray, int]],
    config: StimulusAlignmentConfig,
) -> AlignmentResult:
    """Detect stimuli using MFCC cosine similarity with a sliding window."""
    events = []

    # Compute recording MFCCs
    rec_mfcc = librosa.feature.mfcc(
        y=recording.waveform,
        sr=recording.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )  # shape: (n_mfcc, n_frames)

    for sid, (stim_wav, stim_sr) in sorted(stimulus_audio.items()):
        # Compute stimulus MFCCs
        stim_mfcc = librosa.feature.mfcc(
            y=stim_wav,
            sr=stim_sr,
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )

        stim_len = stim_mfcc.shape[1]
        if stim_len == 0 or rec_mfcc.shape[1] < stim_len:
            continue

        # Compute mean MFCC vector for stimulus (template)
        stim_mean = np.mean(stim_mfcc, axis=1)  # (n_mfcc,)
        stim_norm = np.linalg.norm(stim_mean)
        if stim_norm < 1e-8:
            continue

        # Sliding window with step
        step_frames = max(1, int(
            config.window_step_s * recording.sample_rate / config.hop_length
        ))

        similarities = []
        positions = []

        for i in range(0, rec_mfcc.shape[1] - stim_len + 1, step_frames):
            window_mfcc = rec_mfcc[:, i:i + stim_len]
            window_mean = np.mean(window_mfcc, axis=1)
            window_norm = np.linalg.norm(window_mean)

            if window_norm < 1e-8:
                similarities.append(0.0)
            else:
                cos_sim = float(
                    np.dot(stim_mean, window_mean) / (stim_norm * window_norm)
                )
                similarities.append(cos_sim)
            positions.append(i)

        if not similarities:
            continue

        similarities = np.array(similarities)
        positions = np.array(positions)

        # Find peaks
        min_dist = max(1, int(
            config.peak_distance_s * recording.sample_rate
            / config.hop_length / step_frames
        ))

        peaks, _ = scipy_signal.find_peaks(
            similarities,
            prominence=config.peak_prominence,
            distance=min_dist,
        )

        for peak_idx in peaks:
            confidence = float(similarities[peak_idx])
            if confidence < config.similarity_threshold:
                continue

            frame_idx = positions[peak_idx]
            start_s = float(frame_idx * config.hop_length / recording.sample_rate)
            end_s = float(
                (frame_idx + stim_len) * config.hop_length / recording.sample_rate
            )

            events.append(StimulusEvent(
                sentence_id=sid,
                stimulus_start_s=start_s,
                stimulus_end_s=end_s,
                confidence=confidence,
                method="mfcc_cosine",
            ))

    return AlignmentResult(events=events, method="mfcc_cosine")


# ---------------------------------------------------------------------------
# Method C: Audio Fingerprinting
# ---------------------------------------------------------------------------

def _align_fingerprint(
    recording: AudioData,
    stimulus_audio: Dict[int, Tuple[np.ndarray, int]],
    config: StimulusAlignmentConfig,
) -> AlignmentResult:
    """Detect stimuli using a simplified audio fingerprinting approach.

    Creates spectral peak-based fingerprints and matches them between
    stimulus and recording.
    """
    events = []

    # Generate recording fingerprints
    rec_peaks = _extract_spectral_peaks(
        recording.waveform, recording.sample_rate,
        config.n_fft, config.hop_length, config.fingerprint_freq_bins,
    )
    rec_hashes = _generate_hashes(rec_peaks, config.fingerprint_fan_value)

    for sid, (stim_wav, stim_sr) in sorted(stimulus_audio.items()):
        # Generate stimulus fingerprints
        stim_peaks = _extract_spectral_peaks(
            stim_wav, stim_sr,
            config.n_fft, config.hop_length, config.fingerprint_freq_bins,
        )
        stim_hashes = _generate_hashes(stim_peaks, config.fingerprint_fan_value)

        # Find matching hashes
        offsets = _find_hash_matches(stim_hashes, rec_hashes)

        if not offsets:
            continue

        # Cluster offsets to find alignment points
        stim_duration_s = len(stim_wav) / stim_sr
        matched_times = _cluster_offsets(
            offsets, config.hop_length, recording.sample_rate,
            stim_duration_s, config.min_stimulus_spacing_s,
        )

        for start_s, match_count in matched_times:
            confidence = min(1.0, match_count / max(len(stim_hashes), 1))
            if confidence < config.similarity_threshold:
                continue

            events.append(StimulusEvent(
                sentence_id=sid,
                stimulus_start_s=start_s,
                stimulus_end_s=start_s + stim_duration_s,
                confidence=confidence,
                method="fingerprint",
            ))

    return AlignmentResult(events=events, method="fingerprint")


def _extract_spectral_peaks(
    waveform: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_bins: int,
) -> List[Tuple[int, int]]:
    """Extract spectral peaks (time_frame, freq_bin) from audio."""
    S = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length))

    # Quantize frequency axis
    if S.shape[0] > n_bins:
        # Average frequency bands into bins
        bin_size = S.shape[0] // n_bins
        S_binned = np.zeros((n_bins, S.shape[1]))
        for b in range(n_bins):
            start = b * bin_size
            end = start + bin_size
            S_binned[b] = np.mean(S[start:end], axis=0)
        S = S_binned

    peaks = []
    for t in range(S.shape[1]):
        col = S[:, t]
        if np.max(col) < 1e-8:
            continue
        # Find local maxima in frequency
        freq_peaks, _ = scipy_signal.find_peaks(col, prominence=np.max(col) * 0.1)
        for f in freq_peaks:
            peaks.append((t, int(f)))

    return peaks


def _generate_hashes(
    peaks: List[Tuple[int, int]],
    fan_value: int,
) -> Dict[Tuple[int, int, int], int]:
    """Generate combinatorial hashes from spectral peaks.

    Returns: hash -> time_frame
    """
    hashes = {}
    peaks_sorted = sorted(peaks, key=lambda p: p[0])

    for i, (t1, f1) in enumerate(peaks_sorted):
        for j in range(1, min(fan_value + 1, len(peaks_sorted) - i)):
            t2, f2 = peaks_sorted[i + j]
            dt = t2 - t1
            if dt <= 0 or dt > 200:  # max time delta
                continue
            h = (f1, f2, dt)
            hashes[h] = t1

    return hashes


def _find_hash_matches(
    stim_hashes: Dict[Tuple, int],
    rec_hashes: Dict[Tuple, int],
) -> List[int]:
    """Find time offsets where stimulus hashes match recording hashes."""
    offsets = []
    for h, stim_time in stim_hashes.items():
        if h in rec_hashes:
            rec_time = rec_hashes[h]
            offset = rec_time - stim_time
            offsets.append(offset)
    return offsets


def _cluster_offsets(
    offsets: List[int],
    hop_length: int,
    sr: int,
    stim_duration_s: float,
    min_spacing_s: float,
) -> List[Tuple[float, int]]:
    """Cluster time offsets to find alignment points.

    Returns list of (start_time_s, match_count).
    """
    if not offsets:
        return []

    offsets_s = sorted([o * hop_length / sr for o in offsets])

    # Simple histogram clustering
    bin_size = stim_duration_s * 0.5
    clusters: List[List[float]] = []
    current_cluster: List[float] = [offsets_s[0]]

    for t in offsets_s[1:]:
        if t - current_cluster[-1] < bin_size:
            current_cluster.append(t)
        else:
            clusters.append(current_cluster)
            current_cluster = [t]
    clusters.append(current_cluster)

    # Extract cluster centers
    results = []
    for cluster in clusters:
        center = float(np.median(cluster))
        count = len(cluster)
        if center >= 0:
            results.append((center, count))

    # Enforce minimum spacing
    if len(results) <= 1:
        return results

    filtered = [results[0]]
    for center, count in results[1:]:
        if center - filtered[-1][0] >= min_spacing_s:
            filtered.append((center, count))

    return filtered


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _post_process_events(
    events: List[StimulusEvent],
    config: StimulusAlignmentConfig,
) -> List[StimulusEvent]:
    """Sort events by time, remove duplicates, enforce spacing."""
    if not events:
        return events

    # Sort by time
    events.sort(key=lambda e: e.stimulus_start_s)

    # Remove duplicate detections for same sentence_id (keep highest confidence)
    best_by_sid: Dict[int, StimulusEvent] = {}
    for event in events:
        sid = event.sentence_id
        if sid not in best_by_sid or event.confidence > best_by_sid[sid].confidence:
            best_by_sid[sid] = event

    events = sorted(best_by_sid.values(), key=lambda e: e.stimulus_start_s)

    # Enforce minimum spacing — only merge events with the SAME sentence_id.
    # Different sentence_ids should never replace each other, since consecutive
    # stimuli are legitimately close together (often <5s apart).
    if len(events) <= 1:
        return events

    filtered = [events[0]]
    for event in events[1:]:
        prev = filtered[-1]
        time_gap = event.stimulus_start_s - prev.stimulus_end_s
        if time_gap >= config.min_stimulus_spacing_s:
            filtered.append(event)
        elif event.sentence_id == prev.sentence_id:
            # Same sentence detected twice — keep highest confidence
            if event.confidence > prev.confidence:
                filtered[-1] = event
        else:
            # Different sentence_ids that are close together — keep both
            filtered.append(event)

    return filtered


def _ensure_all_stimuli(
    events: List[StimulusEvent],
    n_expected: int = 30,
    recording_duration_s: float = 0.0,
) -> List[StimulusEvent]:
    """Ensure all expected stimulus events are present.

    If alignment detected fewer than n_expected stimuli, interpolate
    positions for missing ones using detected events as temporal anchors.
    Interpolated events are marked with method='interpolated' and
    confidence=0.0 so downstream stages can identify them.

    Parameters
    ----------
    events : list of StimulusEvent
        Detected stimulus events (may be fewer than n_expected).
    n_expected : int
        Number of expected stimuli (default 30 for EIT).
    recording_duration_s : float
        Total recording duration for boundary clamping.

    Returns
    -------
    list of StimulusEvent
        Exactly n_expected events sorted by stimulus_start_s.
    """
    detected_by_id: Dict[int, StimulusEvent] = {e.sentence_id: e for e in events}

    if len(detected_by_id) >= n_expected:
        return sorted(
            [detected_by_id[sid] for sid in range(1, n_expected + 1)
             if sid in detected_by_id][:n_expected],
            key=lambda e: e.stimulus_start_s,
        )

    detected = sorted(events, key=lambda e: e.stimulus_start_s)

    # Compute average stimulus duration from detected events
    if detected:
        avg_duration = float(np.mean(
            [e.stimulus_end_s - e.stimulus_start_s for e in detected]
        ))
        # Cap at reasonable maximum (stimulus sentences are 1-5s)
        avg_duration = min(avg_duration, 5.0)
    else:
        avg_duration = 3.0

    # Compute average start-to-start interval between consecutive sentence_ids
    if len(detected) >= 2:
        # Use total span divided by sentence_id range for a robust estimate
        first, last = detected[0], detected[-1]
        sid_range = last.sentence_id - first.sentence_id
        if sid_range > 0:
            avg_interval = (
                last.stimulus_start_s - first.stimulus_start_s
            ) / sid_range
        else:
            avg_interval = avg_duration + 7.0
    else:
        avg_interval = avg_duration + 7.0

    logger.info(
        "Ensuring all %d stimuli: %d detected, avg_interval=%.1fs, avg_dur=%.1fs",
        n_expected, len(detected), avg_interval, avg_duration,
    )

    # Build complete list
    all_events: List[StimulusEvent] = []

    for sid in range(1, n_expected + 1):
        if sid in detected_by_id:
            all_events.append(detected_by_id[sid])
            continue

        # Interpolate position from nearest detected anchors
        before = [(e.sentence_id, e.stimulus_start_s)
                  for e in detected if e.sentence_id < sid]
        after = [(e.sentence_id, e.stimulus_start_s)
                 for e in detected if e.sentence_id > sid]

        if before and after:
            b_sid, b_start = before[-1]
            a_sid, a_start = after[0]
            fraction = (sid - b_sid) / (a_sid - b_sid)
            estimated_start = b_start + fraction * (a_start - b_start)
        elif before:
            b_sid, b_start = before[-1]
            estimated_start = b_start + (sid - b_sid) * avg_interval
        elif after:
            a_sid, a_start = after[0]
            estimated_start = max(0, a_start - (a_sid - sid) * avg_interval)
        else:
            estimated_start = (sid - 1) * avg_interval

        estimated_start = max(0.0, estimated_start)
        if recording_duration_s > 0:
            estimated_start = min(estimated_start, max(0.0, recording_duration_s - avg_duration))

        all_events.append(StimulusEvent(
            sentence_id=sid,
            stimulus_start_s=estimated_start,
            stimulus_end_s=estimated_start + avg_duration,
            confidence=0.0,
            method="interpolated",
        ))

    # Sort by time and resolve any temporal inversions
    all_events.sort(key=lambda e: e.stimulus_start_s)

    # Final pass: ensure monotonically increasing start times
    for i in range(1, len(all_events)):
        if all_events[i].stimulus_start_s <= all_events[i - 1].stimulus_end_s:
            # Push this event after the previous one with a small gap
            gap = 1.0
            all_events[i] = StimulusEvent(
                sentence_id=all_events[i].sentence_id,
                stimulus_start_s=all_events[i - 1].stimulus_end_s + gap,
                stimulus_end_s=all_events[i - 1].stimulus_end_s + gap + avg_duration,
                confidence=all_events[i].confidence,
                method=all_events[i].method,
            )

    return all_events
