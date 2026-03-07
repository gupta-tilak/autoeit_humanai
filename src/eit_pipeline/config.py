"""Central configuration for the EIT stimulus-alignment pipeline.

All tuneable parameters are defined here as dataclasses.  Configs are loaded
from YAML, can be overridden programmatically, and are persisted with every
experiment run for reproducibility.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# The 30 target (stimulus) sentences — same for EIT Version 1A and 2A
# ---------------------------------------------------------------------------

TARGET_SENTENCES: List[str] = [
    "Quiero cortarme el pelo",
    "El libro está en la mesa",
    "El carro lo tiene Pedro",
    "El se ducha cada mañana",
    "¿Qué dice usted que va a hacer hoy?",
    "Dudo que sepa manejar muy bien",
    "Las calles de esta ciudad son muy anchas",
    "Puede que llueva mañana todo el día",
    "Las casas son muy bonitas pero caras",
    "Me gustan las películas que acaban bien",
    "El chico con el que yo salgo es español",
    "Después de cenar me fui a dormir tranquilo",
    "Quiero una casa en la que vivan mis animales",
    "A nosotros nos fascinan las fiestas grandiosas",
    "Ella sólo bebe cerveza y no come nada",
    "Me gustaría que el precio de las casas bajara",
    "Cruza a la derecha y después sigue todo recto",
    "Ella ha terminado de pintar su apartamento",
    "Me gustaría que empezara a hacer más calor pronto",
    "El niño al que se le murió el gato está triste",
    "Una amiga mía cuida a los niños de mi vecino",
    "El gato que era negro fue perseguido por el perro",
    "Antes de poder salir él tiene que limpiar su cuarto",
    "La cantidad de personas que fuman ha disminuido",
    "Después de llegar a casa del trabajo tomé la cena",
    "El ladrón al que atrapó la policía era famoso",
    "Le pedí a un amigo que me ayudara con la tarea",
    "El examen no fue tan difícil como me habían dicho",
    "¿Serías tan amable de darme el libro que está en la mesa?",
    "Hay mucha gente que no toma nada para el desayuno",
]


# ---------------------------------------------------------------------------
# Audio file metadata
# ---------------------------------------------------------------------------

@dataclass
class AudioFileConfig:
    """Metadata for a single EIT audio file."""
    participant_id: str
    eit_version: str
    filename: str
    sheet_name: str
    skip_seconds: float = 150.0


DEFAULT_AUDIO_FILES: List[AudioFileConfig] = [
    AudioFileConfig("038010", "2A", "038010_EIT-2A.mp3", "38010-2A", skip_seconds=160.0),
    AudioFileConfig("038011", "1A", "038011_EIT-1A.mp3", "38011-1A", skip_seconds=145.0),
    AudioFileConfig("038012", "2A", "038012_EIT-2A.mp3", "38012-2A", skip_seconds=168.0),
    AudioFileConfig("038015", "1A", "038015_EIT-1A.mp3", "38015-1A", skip_seconds=144.0),
]


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class AudioIOConfig:
    """Audio loading parameters."""
    sample_rate: int = 16_000
    mono: bool = True
    normalise_peak_db: float = -3.0
    noise_reduce: bool = True
    noise_reduce_prop_decrease: float = 0.5
    noise_reduce_stationary: bool = True


@dataclass
class StimulusAlignmentConfig:
    """Stimulus audio matching parameters."""
    # Which matching method to use: cross_correlation | mfcc_cosine | fingerprint
    method: str = "cross_correlation"
    # Enable comparison mode: run all 3 methods and output comparison
    compare_all_methods: bool = False
    # Spectrogram / MFCC parameters
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 20
    # Sliding window parameters (for MFCC cosine)
    window_step_s: float = 0.1
    # Similarity thresholds — lower values detect more stimuli
    # (especially important when stimulus is played from external device at low volume)
    similarity_threshold: float = 0.25
    # Duplicate suppression: minimum time between two detections of the SAME stimulus
    min_stimulus_spacing_s: float = 3.0
    # Peak detection
    peak_prominence: float = 0.15
    peak_distance_s: float = 3.0
    # Expected number of stimuli per recording
    expected_stimuli: int = 30
    # Fingerprinting
    fingerprint_fan_value: int = 15
    fingerprint_freq_bins: int = 64
    # Path to reference stimulus audio files
    stimulus_audio_dir: str = "data/stimulus_audio"


@dataclass
class SilenceDetectionConfig:
    """Silence/speech detection parameters."""
    # VAD backend: silero
    vad_backend: str = "silero"
    # Silero VAD parameters
    silero_threshold: float = 0.5
    silero_min_speech_duration_ms: int = 250
    silero_min_silence_duration_ms: int = 100
    silero_window_size_samples: int = 512
    # Segment filtering
    min_segment_duration_s: float = 0.3
    # Merge speech bursts closer than this (L2 speakers pause mid-sentence)
    merge_gap_s: float = 1.8


@dataclass
class ResponseWindowConfig:
    """Response window construction parameters."""
    min_response_duration_s: float = 0.2
    max_response_duration_s: float = 15.0
    # Padding
    padding_before_s: float = 0.1
    padding_after_s: float = 0.2
    expected_responses: int = 30


@dataclass
class VADRefinementConfig:
    """VAD boundary refinement parameters."""
    expand_window_s: float = 0.5
    # Frame-level VAD for trimming
    frame_duration_ms: int = 30
    vad_threshold: float = 0.5


@dataclass
class DiarizationConfig:
    """Speaker diarization parameters (optional stage)."""
    enabled: bool = False
    hf_token: Optional[str] = None
    num_speakers: int = 2
    min_speakers: int = 2
    max_speakers: int = 2
    model_name: str = "pyannote/speaker-diarization-3.1"
    merge_gap_s: float = 0.8
    min_turn_duration_s: float = 0.3


@dataclass
class ASRConfig:
    """ASR transcription configuration."""
    backend: str = "mlx_whisper"
    model_size: str = "small"
    model_path: Optional[str] = None
    language: str = "es"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    temperature_fallback: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    condition_on_previous_text: bool = False
    word_timestamps: bool = True
    vad_filter: bool = True
    initial_prompt: str = (
        "Transcripción de hablantes no nativos de español repitiendo oraciones. "
        "Incluir vacilaciones y errores tal como se producen. "
        "Ejemplo: Quiero cortarme el pelo. El libro está en la mesa. "
        "Después de cenar me fui a dormir tranquilo."
    )
    low_confidence_threshold: float = -1.0
    multi_pass: bool = True
    second_pass_temperature: float = 0.2
    # MLX specific
    mlx_model_repo: str = "mlx-community/whisper-small-mlx"
    # Batch size for transcription
    batch_size: int = 8


@dataclass
class LeakageCheckConfig:
    """Stimulus leakage check parameters."""
    enabled: bool = True
    similarity_threshold: float = 0.85
    # Library: rapidfuzz or jiwer
    similarity_method: str = "rapidfuzz"


@dataclass
class OutputConfig:
    """Output configuration."""
    export_json: bool = True
    export_csv: bool = True
    export_excel: bool = True
    output_dir: str = "output"


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""
    # Paths
    audio_dir: str = "Sample Audio Files and Transcriptions"
    template_excel: str = "Sample Audio Files and Transcriptions/AutoEIT Sample Audio for Transcribing.xlsx"
    reference_excel: str = "Sample Audio Files and Transcriptions/AutoEIT Sample Transcriptions for Scoring.xlsx"
    output_dir: str = "output"
    experiments_dir: str = "experiments"
    cache_dir: str = ".cache"

    # Sub-configs
    audio_io: AudioIOConfig = field(default_factory=AudioIOConfig)
    stimulus_alignment: StimulusAlignmentConfig = field(default_factory=StimulusAlignmentConfig)
    silence_detection: SilenceDetectionConfig = field(default_factory=SilenceDetectionConfig)
    response_window: ResponseWindowConfig = field(default_factory=ResponseWindowConfig)
    vad_refinement: VADRefinementConfig = field(default_factory=VADRefinementConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    leakage_check: LeakageCheckConfig = field(default_factory=LeakageCheckConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Audio files
    audio_files: List[AudioFileConfig] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_AUDIO_FILES)
    )

    # Experiment metadata
    experiment_name: str = "default"
    experiment_description: str = ""
    tags: List[str] = field(default_factory=list)

    # -------------------------------------------------------------------
    # Serialisation
    # -------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Optional[str] = None) -> str:
        d = self.to_dict()
        text = yaml.dump(
            d, default_flow_style=False, allow_unicode=True, sort_keys=False,
        )
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_json(self, path: Optional[str] = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Recursively build config from nested dict."""
        sub_map = {
            "audio_io": AudioIOConfig,
            "stimulus_alignment": StimulusAlignmentConfig,
            "silence_detection": SilenceDetectionConfig,
            "response_window": ResponseWindowConfig,
            "vad_refinement": VADRefinementConfig,
            "diarization": DiarizationConfig,
            "asr": ASRConfig,
            "leakage_check": LeakageCheckConfig,
            "output": OutputConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, val in d.items():
            if key in sub_map and isinstance(val, dict):
                kwargs[key] = sub_map[key](**val)
            elif key == "audio_files" and isinstance(val, list):
                kwargs[key] = [
                    AudioFileConfig(**af) if isinstance(af, dict) else af
                    for af in val
                ]
            else:
                kwargs[key] = val
        return cls(**kwargs)
