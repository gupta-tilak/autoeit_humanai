"""Configuration management for AutoEIT pipeline.

Centralises all tuneable parameters so that every experiment run is fully
reproducible.  Configs are loaded from YAML, overridden by CLI / code, and
persisted alongside experiment logs.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Audio file metadata
# ---------------------------------------------------------------------------

@dataclass
class AudioFileConfig:
    """Metadata for a single EIT audio file."""
    participant_id: str          # e.g. "038010"
    eit_version: str             # e.g. "2A"
    filename: str                # e.g. "038010_EIT-2A.mp3"
    sheet_name: str              # e.g. "38010-2A"
    skip_seconds: float = 150.0  # intro to skip (seconds)


DEFAULT_AUDIO_FILES: List[AudioFileConfig] = [
    AudioFileConfig("038010", "2A", "038010_EIT-2A.mp3", "38010-2A", skip_seconds=160.0),
    AudioFileConfig("038011", "1A", "038011_EIT-1A.mp3", "38011-1A", skip_seconds=145.0),
    AudioFileConfig("038012", "2A", "038012_EIT-2A.mp3", "38012-2A", skip_seconds=168.0),
    AudioFileConfig("038015", "1A", "038015_EIT-1A.mp3", "38015-1A", skip_seconds=144.0),
]


# ---------------------------------------------------------------------------
# The 30 target (stimulus) sentences — EIT Version A
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
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """Audio pre-processing parameters."""
    sample_rate: int = 16_000
    channels: int = 1
    sample_width: int = 2           # 16-bit PCM
    normalise_peak_db: float = -3.0
    noise_reduce: bool = True
    noise_reduce_prop_decrease: float = 0.5
    noise_reduce_stationary: bool = True


@dataclass
class SegmentationConfig:
    """Audio segmentation parameters."""
    # VAD merging — L2 speakers pause 1.2-1.6s mid-sentence; merge bursts
    # closer than this so a single response isn't split into fragments.
    merge_gap_s: float = 1.8
    # Drop any extracted segment shorter than this (removes split artefacts).
    min_segment_duration_s: float = 1.0
    # Silence detection
    min_silence_len_ms: int = 1500       # min silence between segments
    silence_thresh_db: int = -40         # dBFS threshold for silence
    seek_step_ms: int = 10               # step for silence detection
    # Tone detection
    tone_freq_hz: float = 1000.0         # expected tone frequency
    tone_min_duration_ms: int = 100      # minimum duration of a tone
    tone_max_duration_ms: int = 2000     # maximum duration of a tone
    tone_detection_enabled: bool = True
    # Response extraction
    min_response_duration_ms: int = 200  # discard very short blips
    max_response_duration_ms: int = 15000  # cap on response length
    # Expected counts
    expected_segments: int = 30
    # Padding around detected response
    padding_before_ms: int = 100
    padding_after_ms: int = 200


@dataclass
class ASRConfig:
    """ASR backend configuration — model-agnostic."""
    backend: str = "whisper_cpp"         # whisper_cpp | mlx_whisper | faster_whisper
    model_size: str = "small"            # tiny | base | small | medium | large
    model_path: Optional[str] = None     # path to custom / fine-tuned model
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
    # Confidence thresholds
    low_confidence_threshold: float = -1.0   # avg log-prob trigger
    # Multi-pass
    multi_pass: bool = True
    second_pass_temperature: float = 0.2
    # whisper.cpp specific
    whisper_cpp_bin: Optional[str] = None      # path to whisper.cpp main binary
    whisper_cpp_model_path: Optional[str] = None  # path to .bin ggml model
    whisper_cpp_use_gpu: bool = True            # use Metal on macOS
    whisper_cpp_threads: int = 4
    # mlx-whisper specific
    mlx_model_repo: str = "mlx-community/whisper-small-mlx"


@dataclass
class PostProcessingConfig:
    """Post-processing rules."""
    fix_accents: bool = True
    remove_hallucinations: bool = True
    hallucination_repeat_threshold: int = 3  # repeated n-gram size
    format_disfluencies: bool = True
    no_response_energy_threshold: float = 0.01  # RMS below → [no response]
    no_response_min_words: int = 0


@dataclass
class DiarizationConfig:
    """Speaker diarization parameters (pyannote-based, Phase 2)."""
    enabled: bool = False                        # opt-in — requires HF token
    hf_token: Optional[str] = None               # HF token or set HUGGINGFACE_TOKEN env var
    num_speakers: int = 2                        # 0 = auto-detect
    min_speakers: int = 2                        # EIT is always exactly 2 speakers
    max_speakers: int = 2                        # EIT is always exactly 2 speakers
    merge_gap_s: float = 0.8                     # non-native speakers pause more mid-sentence
    min_turn_duration_s: float = 0.3             # discard shorter turns
    window_after_tone_s: float = 2.0             # window for voting on response speaker
    model_name: str = "pyannote/speaker-diarization-3.1"


@dataclass
class EvaluationConfig:
    """Self-evaluation parameters."""
    compute_self_consistency: bool = True
    consistency_temperatures: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4]
    )
    compute_stimulus_similarity: bool = True
    report_confidence_scores: bool = True


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
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    postprocessing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Audio files
    audio_files: List[AudioFileConfig] = field(default_factory=lambda: copy.deepcopy(DEFAULT_AUDIO_FILES))

    # Experiment metadata
    experiment_name: str = "default"
    experiment_description: str = ""
    tags: List[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Optional[str] = None) -> str:
        d = self.to_dict()
        text = yaml.dump(d, default_flow_style=False, allow_unicode=True, sort_keys=False)
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
        """Recursively reconstruct dataclass from dict."""
        if "preprocessing" in d and isinstance(d["preprocessing"], dict):
            d["preprocessing"] = PreprocessingConfig(**d["preprocessing"])
        if "segmentation" in d and isinstance(d["segmentation"], dict):
            d["segmentation"] = SegmentationConfig(**d["segmentation"])
        if "asr" in d and isinstance(d["asr"], dict):
            d["asr"] = ASRConfig(**d["asr"])
        if "postprocessing" in d and isinstance(d["postprocessing"], dict):
            d["postprocessing"] = PostProcessingConfig(**d["postprocessing"])
        if "diarization" in d and isinstance(d["diarization"], dict):
            d["diarization"] = DiarizationConfig(**d["diarization"])
        if "evaluation" in d and isinstance(d["evaluation"], dict):
            d["evaluation"] = EvaluationConfig(**d["evaluation"])
        if "audio_files" in d and isinstance(d["audio_files"], list):
            d["audio_files"] = [
                AudioFileConfig(**af) if isinstance(af, dict) else af
                for af in d["audio_files"]
            ]
        return cls(**d)

    def resolve_paths(self, base_dir: str) -> "PipelineConfig":
        """Resolve relative paths against *base_dir*."""
        cfg = copy.deepcopy(self)
        base = Path(base_dir)
        cfg.audio_dir = str(base / cfg.audio_dir)
        cfg.template_excel = str(base / cfg.template_excel)
        cfg.reference_excel = str(base / cfg.reference_excel)
        cfg.output_dir = str(base / cfg.output_dir)
        cfg.experiments_dir = str(base / cfg.experiments_dir)
        cfg.cache_dir = str(base / cfg.cache_dir)
        return cfg
