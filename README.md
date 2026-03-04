# AutoEIT — Automated Elicited Imitation Task Transcription

A model-agnostic, Mac-native optimized pipeline for automatically transcribing non-native Spanish speaker responses from Elicited Imitation Task (EIT) audio recordings.

## Architecture

```
MP3 → Preprocessing → Segmentation → ASR → Post-Processing → Excel Output
       │                │               │          │
  Normalize audio   30 segments     Spanish-tuned  Preserve learner errors
  Noise reduction   per file        multi-pass     Fix ASR artifacts only
  Skip intro        tone detection  prompt-guided  Format disfluencies
```

### Key Design Principles

- **Model-agnostic ASR layer:** Switch between whisper.cpp (Metal GPU), MLX-Whisper (Apple native), faster-whisper (CTranslate2), or custom fine-tuned models via a single config change
- **Mac-native optimization:** Metal acceleration via whisper.cpp; MLX framework for unified memory inference on Apple Silicon
- **Experiment tracking:** Every pipeline run is logged with full config snapshot, per-sentence results, confidence scores, and evaluation metrics — enabling systematic iteration
- **Disfluency preservation:** Transcribes exact participant production including false starts, pauses, and errors — never corrects grammar/vocabulary

## Project Structure

```
autoeit_humanai/
├── src/autoeit/                 # Core pipeline package
│   ├── __init__.py
│   ├── config.py                # Configuration management (YAML-backed)
│   ├── experiment_logger.py     # Experiment tracking & comparison
│   ├── pipeline.py              # Main orchestrator
│   ├── preprocessing.py         # Audio preprocessing (normalization, noise reduction)
│   ├── segmentation.py          # 30-segment extraction (tone detection + hybrid)
│   ├── postprocessing.py        # ASR error correction (preserving learner errors)
│   ├── evaluation.py            # Self-evaluation (confidence, similarity, style)
│   ├── output.py                # Excel output generation
│   ├── __main__.py              # CLI entry point
│   └── asr/                     # Model-agnostic ASR backends
│       ├── base.py              # Abstract interface
│       ├── factory.py           # Backend factory + custom registration
│       ├── whisper_cpp.py       # whisper.cpp (Metal accelerated)
│       ├── mlx_whisper.py       # MLX-Whisper (Apple Silicon native)
│       └── faster_whisper.py    # faster-whisper (CTranslate2)
├── configs/                     # YAML configuration files
│   ├── default.yaml             # MLX-Whisper small (recommended)
│   └── whisper_cpp.yaml         # whisper.cpp with Metal
├── experiments/                 # Experiment logs (auto-generated)
├── output/                      # Pipeline output (auto-generated)
├── AutoEIT_Pipeline.ipynb       # Main Jupyter notebook
├── requirements.txt             # Core dependencies
├── requirements-mlx.txt         # MLX-Whisper dependencies
├── requirements-whispercpp.txt  # whisper.cpp dependencies
├── requirements-faster-whisper.txt
└── pyproject.toml               # Package configuration
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Install ASR backend (choose one):
# Option A: MLX-Whisper (recommended for Apple Silicon)
pip install -r requirements-mlx.txt

# Option B: whisper.cpp 
pip install -r requirements-whispercpp.txt

# Option C: faster-whisper (works on all platforms)
pip install -r requirements-faster-whisper.txt

# Install package in development mode
pip install -e .
```

### 2. Run via Jupyter Notebook

```bash
jupyter notebook AutoEIT_Pipeline.ipynb
```

### 3. Run via CLI

```bash
# Default config (MLX-Whisper small)
python -m autoeit run

# Specific config
python -m autoeit run --config configs/whisper_cpp.yaml

# Override backend/model
python -m autoeit run --backend mlx_whisper --model-size medium

# Single participant
python -m autoeit run --participants 038010

# Name your experiment
python -m autoeit run --name "medium_model_v2" --tags medium experiment
```

### 4. Run Programmatically

```python
from autoeit.config import PipelineConfig
from autoeit.pipeline import AutoEITPipeline

config = PipelineConfig.from_yaml('configs/default.yaml')
config.asr.backend = 'mlx_whisper'
config.asr.model_size = 'medium'

pipeline = AutoEITPipeline(config, base_dir='.')
run = pipeline.run(name='my_experiment')
```

## Switching ASR Backends

The architecture is designed so that switching models is trivial:

```yaml
# In configs/default.yaml, change one line:
asr:
  backend: "mlx_whisper"       # or "whisper_cpp" or "faster_whisper"
  model_size: "small"          # tiny | base | small | medium | large
  model_path: null             # path to custom/fine-tuned model
```

### Using a Fine-Tuned Model

```python
from autoeit.config import PipelineConfig

config = PipelineConfig()
config.asr.backend = 'mlx_whisper'
config.asr.model_path = '/path/to/my-finetuned-model'
```

### Registering a Custom Backend

```python
from autoeit.asr.factory import register_backend

register_backend("my_custom_asr", "mypackage.backend.MyASRBackend")

config.asr.backend = "my_custom_asr"
```

## Experiment Tracking

Every pipeline run is automatically logged:

```bash
# List all runs
python -m autoeit list-runs

# Compare runs
python -m autoeit compare --runs 20260304_123456_baseline 20260305_091011_medium
```

Each experiment stores:
- **Full config snapshot** (YAML + JSON)
- **Per-sentence transcription results** with confidence scores
- **Raw vs post-processed transcriptions**
- **Evaluation metrics** (confidence, stimulus similarity, quality scores)
- **Human-readable summary**

```
experiments/
└── 20260304_123456_baseline/
    ├── run.json                           # Complete experiment data
    ├── config.json                        # Config snapshot
    ├── summary.txt                        # Human-readable report
    ├── 038010_2A_transcriptions.json      # Per-participant results
    ├── 038011_1A_transcriptions.json
    ├── 038012_2A_transcriptions.json
    └── 038015_1A_transcriptions.json
```

## Audio Processing Details

| File | Participant | Version | Skip Intro | Duration |
|------|-------------|---------|------------|----------|
| 038010_EIT-2A.mp3 | 038010 | 2A | 2:30 | ~8.9 MB |
| 038011_EIT-1A.mp3 | 038011 | 1A | 2:30 | ~9.1 MB |
| 038012_EIT-2A.mp3 | 038012 | 2A | **12:00** | ~18.2 MB |
| 038015_EIT-1A.mp3 | 038015 | 1A | 2:30 | ~8.6 MB |

### Segmentation Strategy

1. **Tone detection** — Find ~1kHz beeps between stimulus and response
2. **Speech region detection** — Energy-based non-silence detection
3. **Classification** — Map speech regions to stimulus/response using tone positions
4. **Validation** — Ensure exactly 30 response segments

### Post-Processing Rules

Preserves learner errors while fixing ASR artifacts:

| Fix (ASR error) | Preserve (learner error) |
|-----------------|-------------------------|
| Missing accents on known words | Gender agreement errors |
| Whisper hallucinations | Verb conjugation errors |
| Repeated n-grams | Word substitutions |
| Noise artifact text | Word order changes |
| | Missing/added words |

## System Requirements

- **macOS** on Apple Silicon (M1/M2/M3) recommended
- Python 3.9+
- ffmpeg (`brew install ffmpeg`)
- 8GB+ RAM (16GB recommended for medium model)
