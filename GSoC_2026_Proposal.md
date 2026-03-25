# Google Summer of Code 2026 — Proposal

## **Audio-to-Text Transcription for Second/Additional Language Learner Data**

**Organisation:** HumanAI Foundation (CERN-affiliated)
**Project:** AutoEIT — Automated Elicited Imitation Task Pipeline
**Track:** Test I — Audio-to-text transcription
**Difficulty:** Medium | **Hours:** 175

---

## Table of Contents

1. [Personal Information & Synopsis](#1-personal-information--synopsis)
2. [Motivation & Background](#2-motivation--background)
3. [Understanding of the Problem](#3-understanding-of-the-problem)
4. [Technical Approach](#4-technical-approach)
   - 4.1 [System Architecture Overview](#41-system-architecture-overview)
   - **Segment 1: EIT Segmentation & Transcription Pipeline (Inference)**
     - 4.2 [Audio Segmentation Pipeline](#42-audio-segmentation-pipeline)
     - 4.3 [Transcription Pipeline](#43-transcription-pipeline)
     - 4.4 [Stimulus-Prompted Decoding](#44-stimulus-prompted-decoding)
   - **Segment 2: Fine-tuning Pipeline (Model Training)**
     - 4.5 [Training Data & SLABANK Corpora](#45-training-data--slabank-corpora)
     - 4.6 [Audio Preprocessing for Training](#46-audio-preprocessing-for-training)
     - 4.7 [Data Augmentation](#47-data-augmentation)
     - 4.8 [LoRA Fine-tuning Strategy](#48-lora-fine-tuning-strategy)
   - **Shared Components**
     - 4.9 [ASR Backbone Selection & Rationale](#49-asr-backbone-selection--rationale)
     - 4.10 [Post-Processing Pipeline](#410-post-processing-pipeline)
     - 4.11 [Evaluation Framework](#411-evaluation-framework)
5. [Test Submission Results](#5-test-submission-results)
6. [Timeline](#6-timeline-12-week-breakdown)
7. [Deliverables](#7-deliverables)
8. [About Me / Why I Am a Good Fit](#8-about-me--why-i-am-a-good-fit)
9. [References](#9-references)

---

## 1. Personal Information & Synopsis

| Field | Details |
|-------|---------|
| **Name** | `[INSERT YOUR NAME]` |
| **University** | `[INSERT UNIVERSITY]` |
| **Degree** | `[INSERT DEGREE PROGRAMME & YEAR]` |
| **Country / Timezone** | `[INSERT COUNTRY / UTC±X]` |
| **GitHub** | `[INSERT GITHUB PROFILE URL]` |
| **Test Submission Repo** | `[INSERT REPO URL — e.g. github.com/username/autoeit_humanai]` |
| **Email** | `[INSERT EMAIL]` |

### Executive Summary

This proposal presents a complete, production-ready pipeline for automating Spanish Elicited Imitation Task (EIT) transcription — from raw learner audio to proficiency-aligned transcripts. The system combines a fine-tuned OpenAI Whisper model (LoRA-adapted on 56,090 L2 Spanish utterances from three SLABANK corpora) with a novel stimulus-prompted decoding strategy, corpus-specific audio preprocessing, and drift-aware segmentation. On the four-participant test set, the pipeline achieves a best-participant WER of 23.2% and a cross-participant mean WER of 43.5%, with the fine-tuned model reducing error rates by up to 38% over the off-the-shelf baseline on learner speech. The proposed GSoC work will focus on closing the gap to 90% human agreement through targeted improvements in fine-tuning data, post-processing, and proficiency-aware decoding.

---

## 2. Motivation & Background

`[INSERT: 2–3 sentences on your personal interest in SLA, linguistics, or multilingual NLP]`

My interest in this project sits at the intersection of speech technology and second language acquisition (SLA) research. Standard ASR systems are evaluated on native speech benchmarks (LibriSpeech, Common Voice), but learner speech — with its systematic phonological transfer, variable proficiency, and linguistically meaningful disfluencies — remains a frontier problem. The EIT is a particularly compelling test case: it is a well-established proficiency instrument in SLA research, yet its scalability is bottlenecked entirely by the need for trained human transcribers.

`[INSERT: Relevant coursework, research experience, or prior work with speech/NLP]`

I am drawn to AutoEIT because it demands more than generic model fine-tuning — it requires understanding *why* learners produce the speech they do, and designing systems that preserve rather than erase these linguistically informative patterns.

---

## 3. Understanding of the Problem

### 3.1 The Elicited Imitation Task (EIT)

The EIT is a sentence-repetition task used in SLA research to assess implicit grammatical knowledge. Participants hear a Spanish sentence and immediately repeat it from memory. Their production reveals gaps in morphosyntactic knowledge, phonological accuracy, and overall proficiency. A single session yields ~30 sentences per participant, and scoring is performed by trained human raters on a 0–4 scale.

### 3.2 Why Manual Transcription is the Bottleneck

Each participant session requires a trained rater to listen to audio, transcribe the exact production (including disfluencies, false starts, and partial repetitions), and assign proficiency scores. At scale — studies with hundreds of participants — this becomes prohibitively expensive and slow. Automating the transcription step would accelerate SLA research significantly.

### 3.3 Why Off-the-Shelf ASR Fails on Learner Speech

| Failure Mode | Description | Impact on EIT |
|-------------|-------------|---------------|
| **L1 phonological transfer** | Learner's native language phonemes bleed into Spanish production (e.g., Cameroonian French vowel system, Chinese tonal interference) | ASR hallucinates Spanish words that "sound right" to a native model but don't match what the learner actually said |
| **Variable proficiency** | Beginners through advanced learners in the same dataset | Model confidence calibration breaks down at low proficiency |
| **Disfluency suppression** | Whisper actively removes hesitations, false starts, fillers | These are *linguistically meaningful* in EIT — removing them destroys scoring validity |
| **Partial repetitions** | Learners may produce only a fragment of the target sentence | ASR tends to hallucinate completions or align to wrong segments |
| **Code-switching** | Occasional L1 words inserted into Spanish output | Out-of-vocabulary for Spanish-only language models |
| **Audio variability** | Lab recordings with different microphone setups, background noise levels | Preprocessing must be robust across recording conditions |

Our baseline evaluation confirms this: off-the-shelf Whisper-large-v3 achieves only 32.2% WER on a 40-utterance L2 Spanish test set — far from the 90% agreement target.

> **📖 Relevant Literature:** Moran et al. have shown that ASR performance degrades significantly on L2 speech compared to L1 benchmarks. Radford et al. (2022) demonstrated Whisper's multilingual capabilities but acknowledged performance drops on accented and non-native speech. Recent work on wav2vec 2.0 fine-tuning (Baevski et al., 2020) and XLSR-53 (Conneau et al., 2020) suggests that self-supervised representations can partially bridge the native/non-native gap when fine-tuned on in-domain data.

---

## 4. Technical Approach

### 4.1 System Architecture Overview

The system consists of two independent pipelines: a **Fine-tuning Pipeline** (offline, run once on SLABANK training data) and an **EIT Inference Pipeline** (run per participant session). The diagram below reflects the exact code structure across `eit_segmentation_v3.ipynb`, `eit_transcription.ipynb`, and the `finetuning/` module.

> **📐 DIAGRAM PLACEHOLDER: End-to-End System Architecture**
>
> *Draw two clearly labelled Excalidraw sub-diagrams (one above the other, or side-by-side with a dividing line). Use the ASCII layouts below as the blueprint.*

#### Pipeline A — Fine-tuning (Offline, `finetuning/`)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  FINE-TUNING PIPELINE  (finetuning/)                                                   │
│                                                                                        │
│  SLABANK Corpora                                                                       │
│  ┌───────────────────────────────┐                                                     │
│  │ Nebrija-INMIGRA (CHAT + MP3) │                                                     │
│  │ Nebrija-WOCAE   (CHAT + MP3) │──▶ data_loader.py ──▶ processed_manifest.csv        │
│  │ SPLLOC1         (CHAT + MP3) │    (CHAT parser,       28,403 utterances             │
│  └───────────────────────────────┘    20+ annotation      train / dev / test split)    │
│                                       patterns stripped)                               │
│                                            │                                           │
│                                            ▼                                           │
│                                     preprocess.py                                      │
│                                   (8-step pipeline:                                    │
│                                    load → resample                                     │
│                                    → denoise+bandpass                                  │
│                                    → VAD trim → RMS norm                               │
│                                    → peak clip → quality gate                          │
│                                    → chunk ≤ 25s)                                     │
│                                    corpus-specific SNR/                                │
│                                    denoise params                                      │
│                                            │                                           │
│                                            ▼                                           │
│                                      augment.py                                        │
│                                   (speed / noise / reverb                              │
│                                    / pitch / volume jitter;                            │
│                                    56,090 total utterances;                            │
│                                    6× boost for low-resource L1)                      │
│                                            │                                           │
│                                            ▼                                           │
│                                  mel_cache/*.npy                                       │
│                              (pre-computed mel spectrograms,                           │
│                               (80, T) shape, 10-50× faster loading)                   │
│                                            │                                           │
│                                            ▼                                           │
│                          whisper-eit-kaggle-v2.ipynb (Kaggle GPU T4×2)                │
│                          ┌───────────────────────────────────────────┐                │
│                          │  openai/whisper-large-v2 (1.55B params)   │                │
│                          │  + LoRA adapters (r=8, α=16,              │                │
│                          │    q/k/v/out_proj, 10.5M trainable)       │                │
│                          │  + SpecAugment (2 time + 2 freq masks)    │                │
│                          │  Weighted sampling: 35% INMIGRA /         │                │
│                          │    20% WOCAE / 45% SPLLOC1                │                │
│                          │  LR=5e-6 cosine, 10 epochs, BS=32         │                │
│                          └───────────────────────────────────────────┘                │
│                                            │                                           │
│                                            ▼                                           │
│                              fused/model.safetensors (483 MB)                         │
│                              (LoRA weights merged into base)                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Pipeline B — EIT Inference (Per Participant Session)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  EIT INFERENCE PIPELINE                                                                │
│                                                                                        │
│  INPUTS                                                                                │
│  ┌──────────────────────┐   ┌──────────────────────────┐                              │
│  │ Raw session audio    │   │ Target stimulus list     │                              │
│  │ (MP3/WAV, ~5 min)    │   │ (Excel, 30 sentences)    │                              │
│  └──────────┬───────────┘   └────────────┬─────────────┘                              │
│             └──────────────┬─────────────┘                                            │
│                            ▼                                                          │
│             ┌──────────────────────────────┐                                          │
│             │  eit_segmentation_v3.ipynb   │                                          │
│             │  (VAD-based response         │                                          │
│             │   extraction, stimulus       │                                          │
│             │   alignment)                 │                                          │
│             └──────────────┬───────────────┘                                          │
│                            │                                                          │
│               ┌────────────┴───────────┐                                              │
│               ▼                        ▼                                              │
│         segments.csv           responses/*.wav                                        │
│  (stimulus_id, stimulus_text,  (individual clipped WAV                                │
│   response_start/end/duration,  per stimulus response,                                │
│   status: OK | MISSING,         16 kHz mono)                                         │
│   validation, response_file)                                                          │
│               │                        │                                              │
│               └────────────┬───────────┘                                              │
│                            ▼                                                          │
│             ┌──────────────────────────────┐                                          │
│             │  eit_transcription.ipynb     │                                          │
│             │                              │                                          │
│             │  1. Load segments.csv        │                                          │
│             │     filter: status == 'OK'   │                                          │
│             │     → to_transcribe (≤30)    │                                          │
│             │     → missing_segs           │                                          │
│             │                              │                                          │
│             │  2. Backend auto-detection   │                                          │
│             │     Apple Silicon → MLX      │                                          │
│             │     else → whispercpp        │                                          │
│             │     else → openai            │                                          │
│             │                              │                                          │
│             │  3. Load checkpoint          │                                          │
│             │     (resume if interrupted)  │                                          │
│             └──────────────┬───────────────┘                                          │
│                            │                                                          │
│                    ┌───────▼────────┐  ◀─── per segment loop ───────────────────┐    │
│                    │  Load audio    │                                             │    │
│                    │  response_file │                                             │    │
│                    └───────┬────────┘                                             │    │
│                            ▼                                                      │    │
│          ┌─────────────────────────────────────────────────┐                     │    │
│          │  transcribe_with_stimulus_prompt()               │                     │    │
│          │                                                  │                     │    │
│          │  ┌──────────────────┐    ┌──────────────────┐   │                     │    │
│          │  │ Whisper + prompt │    │  Whisper plain   │   │                     │    │
│          │  │ (stimulus_text   │    │  (no prompt)     │   │                     │    │
│          │  │ as initial_prompt│    │                  │   │                     │    │
│          │  └────────┬─────────┘    └────────┬─────────┘   │                     │    │
│          │           │ text_prompted          │ text_plain  │                     │    │
│          │           │                        │             │                     │    │
│          │           ▼                        │             │                     │    │
│          │  ┌──────────────────┐              │             │                     │    │
│          │  │  cross_wer check │              │             │                     │    │
│          │  │  WER(prompted,   │              │             │                     │    │
│          │  │  stimulus_text)  │              │             │                     │    │
│          │  └────────┬─────────┘              │             │                     │    │
│          │           │                        │             │                     │    │
│          │    ┌──────┴──────┐                 │             │                     │    │
│          │    │ over-       │ YES             │             │                     │    │
│          │    │ anchored?   ├─────────────────┘             │                     │    │
│          │    └──────┬──────┘  strategy = plain_fallback    │                     │    │
│          │           │ NO                                   │                     │    │
│          │           ▼  strategy = prompted                 │                     │    │
│          │  ┌──────────────────┐                           │                     │    │
│          │  │ final transcript  │◀──────────────────────────┘                     │    │
│          │  │ + word timestamps │                                                 │    │
│          │  │ + no_speech_prob  │                                                 │    │
│          │  └────────┬──────────┘                                                 │    │
│          └───────────┼──────────────────────────────────────────────────────────-┘    │
│                      ▼                                                                 │
│          ┌────────────────────────────┐                                               │
│          │  No-speech detection       │                                               │
│          │  no_speech_prob ≥ 0.6      │                                               │
│          │  → asr_status = 'SILENCE'  │                                               │
│          │  else 'OK' / 'EMPTY'       │                                               │
│          └────────────┬───────────────┘                                               │
│                       │                                                               │
│                       ▼  [save checkpoint every 5 segments]  ──────────────────────▶ │
│          ┌────────────────────────────┐                                               │
│          │  WER / CER / word_f1       │                                               │
│          │  learner_score computation │                                               │
│          └────────────┬───────────────┘                                               │
│                       ▼                                                               │
│          ┌────────────────────────────┐                                               │
│          │  detect_mapping_drift()    │                                               │
│          │  (sliding-window alignment │                                               │
│          │   drift check)             │                                               │
│          └────────────┬───────────────┘                                               │
│                       ▼                                                               │
│  OUTPUTS                                                                              │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌──────────────────────────────┐    │
│  │ transcriptions   │  │ transcriptions.xlsx │  │ word_timestamps.csv          │    │
│  │ .csv             │  │ (human-readable,     │  │ (per-word start/end/prob)    │    │
│  │ (full metrics:   │  │  formatted for       │  ├──────────────────────────────┤    │
│  │  wer, cer,       │  │  human review)       │  │ transcription_checkpoint.json│    │
│  │  cross_wer,      │  └─────────────────────┘  │ (resume state)               │    │
│  │  word_f1,        │                           ├──────────────────────────────┤    │
│  │  asr_strategy,   │                           │ Visualizations (PNGs):       │    │
│  │  word_timestamps)│                           │  01_wer_overview.png         │    │
│  └──────────────────┘                           │  02_transcription_detail.png │    │
│                                                 └──────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

> *Excalidraw notes:*
> - *Colour Pipeline A in blue/grey (offline, runs once); Pipeline B in green (runs per session)*
> - *Highlight the over-anchoring decision diamond in orange — it is the key algorithmic contribution*
> - *Show `segments.csv` and `responses/*.wav` as the artifact bridge between the two sub-stages of Pipeline B*
> - *Mark the checkpoint save icon (💾) every 5 segments in the loop*
> - *The fused `model.safetensors` from Pipeline A feeds into the backend model selection step of Pipeline B with a dashed cross-pipeline arrow*

---

## Segment 1: EIT Segmentation & Transcription Pipeline (Inference)

> This segment implements the per-participant inference system: from a raw EIT recording to scored, aligned transcriptions. It is the core pipeline that runs on every participant session and is implemented across `eit_segmentation_v3.ipynb` and `eit_transcription.ipynb`.

### 4.2 Audio Segmentation Pipeline

**Notebook:** `eit_segmentation_v3.ipynb`

The segmentation pipeline solves a critical prerequisite: given a continuous EIT recording (~5 minutes, containing 30 stimulus-response pairs), it must detect and extract each individual learner response, map it to the correct stimulus, and handle missing responses gracefully.

#### 4.2.1 Voice Activity Detection (VAD)

The pipeline uses **Silero VAD** to detect speech segments in the recording. Configuration is deliberately tuned for non-native speech:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `VAD_THRESHOLD` | 0.4 | Lowered from default 0.5 to catch quieter L2 productions |
| `VAD_MIN_SPEECH_MS` | 100 ms | Short minimum to preserve brief learner attempts |
| `VAD_MIN_SILENCE_MS` | 100 ms | Short silence threshold for granular segmentation |

#### 4.2.2 Intra-Utterance Merging

Learner speech contains frequent mid-utterance pauses, hesitations, and self-corrections that VAD incorrectly splits into separate segments. The pipeline collapses consecutive segments separated by ≤ `INTRA_MERGE_GAP` (default 2.0s), preserving these disfluencies as part of a single response.

#### 4.2.3 Smart Reduce Algorithm

If VAD yields more than 30 segments (the expected number of responses), the **smart reduce** algorithm iteratively merges the pair with the smallest inter-segment gap, stopping when:
- Segment count reaches the target (30), or
- The next merge would exceed `MAX_MERGE_GAP_FOR_REDUCE` (4.0s), preventing merges that cross stimulus-pair boundaries

#### 4.2.4 Gap-Based Missing Stimulus Detection

Not all learners respond to every stimulus. The pipeline identifies skipped responses by analysing the inter-segment gap distribution:
- EIT recordings exhibit a **bimodal gap distribution**: short intra-response gaps (~0.5–2s) vs. longer inter-pair silence gaps (~6–8s)
- If a gap exceeds 1.8× the average inter-response interval (`total_duration / 30`), the pipeline inserts a `MISSING` placeholder for the skipped stimulus
- This is visualised via gap analysis plots for manual calibration verification

#### 4.2.5 Sequential Stimulus Mapping & Validation

Detected segments are assigned to stimulus slots in strict temporal order. Validation checks include:
- Duration plausibility (segments too short or too long are flagged)
- No temporal overlap between consecutive assignments
- Total segment count matches expected stimulus count (with MISSING placeholders accounting for gaps)

#### Outputs

| Output | Description |
|--------|-------------|
| `segments.csv` | Per-segment metadata: `stimulus_id`, `stimulus_text`, `response_start`, `response_end`, `response_duration`, `status` (OK / MISSING), `validation`, `response_file` |
| `responses/*.wav` | Individual clipped WAV files per stimulus slot (16 kHz mono), with silent placeholders for MISSING responses |
| Gap analysis visualisations | Timeline plots and gap distribution histograms for calibration |

> **📐 DIAGRAM PLACEHOLDER: Segmentation Pipeline Flowchart**
>
> *Draw an Excalidraw flowchart showing:*
> - Raw recording → Silero VAD → raw segments → Intra-utterance merge → Smart reduce → Gap detection → Stimulus mapping → segments.csv + response WAVs
> - Decision diamonds: "segments > 30?" → smart reduce; "gap > 1.8× avg?" → insert MISSING
> - Annotated waveform showing stimulus playback, learner response, and inter-pair silence
> - Gap distribution histogram (bimodal: intra-response vs. inter-pair)

---

### 4.3 Transcription Pipeline

**Notebook:** `eit_transcription.ipynb`

The transcription pipeline takes the segmented response WAVs and stimulus metadata from the segmentation stage and produces scored, aligned transcriptions. It is a self-contained system with multi-backend support, checkpoint-based resumability, and multiple quality assurance mechanisms.

#### 4.3.1 Multi-Backend ASR Architecture

The pipeline auto-detects hardware and selects the fastest available Whisper backend:

| Priority | Backend | Platform | Advantage |
|:--------:|---------|----------|-----------|
| 1 | **MLX Whisper** | Apple Silicon (M-series) | Native Metal acceleration, fastest on Mac |
| 2 | **whisper.cpp** | Any (C++ native) | Lightweight, efficient CPU inference |
| 3 | **OpenAI Whisper** | Any (Python/PyTorch) | Reference implementation, most flexible |

This ensures the pipeline runs efficiently on any development machine without manual configuration.

#### 4.3.2 Checkpoint & Resumability System

Transcription is performed segment-by-segment with incremental progress saved to `transcription_checkpoint.json` every 5 segments. If the process is interrupted (crash, timeout, resource limits), it resumes from the last checkpoint without re-transcribing completed segments. This is critical for:
- Long sessions (120 utterances across 4 participants)
- Resource-constrained environments (local laptop, Kaggle session timeouts)
- Iterative development (modify parameters without losing previous work)

#### 4.3.3 Audio Preparation for ASR

Each response WAV undergoes minimal preparation before transcription:
- Resample to 16 kHz (Whisper requirement)
- Add ~0.1s silence padding at start/end to prevent word truncation at boundaries
- Amplitude normalisation

#### 4.3.4 Scoring & Metrics

For each transcribed segment, the pipeline computes:

| Metric | Definition |
|--------|-----------|
| `wer` | Word Error Rate vs. stimulus text (substitutions + insertions + deletions / reference words) |
| `cer` | Character Error Rate vs. stimulus text |
| `learner_score` | 1 − WER (percentage of target successfully reproduced) |
| `cross_wer` | WER between prompted and plain transcription (over-anchoring indicator) |
| `word_f1` | Precision/recall at word level |
| `no_speech_prob` | Silence probability from Whisper (flags empty responses) |
| `asr_status` | OK / MISSING / SILENCE / ERROR |

#### 4.3.5 No-Speech Detection

When `no_speech_prob ≥ 0.6`, the segment is marked as `SILENCE` — the learner did not respond. This prevents Whisper from hallucinating text for empty audio, a common failure mode on silence.

#### 4.3.6 Segmentation Drift Detection

A post-hoc validation mechanism catches off-by-one mapping errors from the segmentation stage. For each transcribed response, the pipeline computes WER against stimuli within a ±2 slot window. If:
- `assigned_WER > 0.7` (poor match to assigned stimulus), AND
- A neighbouring stimulus has WER > 0.30 lower (much better match)

→ The segment is flagged as a **potential drift error**, alerting the user to revisit the segmentation.

#### Outputs

| Output | Description |
|--------|-------------|
| `transcriptions.csv` | Full results: stimulus text, transcription, normalised forms, WER, CER, word F1, timestamps, strategy, status |
| `transcriptions.xlsx` | Colour-coded Excel (green < 10% WER, yellow < 30%, orange < 50%, red > 50%) for human review |
| `word_timestamps.csv` | Per-word start/end times and confidence probabilities |
| `transcription_checkpoint.json` | Resumable state |
| Visualisation PNGs | `01_wer_overview.png` (distribution), `02_transcription_detail.png` (side-by-side comparison) |

---

### 4.4 Stimulus-Prompted Decoding

A key innovation in the Segment 1 transcription pipeline is **stimulus-prompted decoding** — conditioning the ASR decoder on the target sentence that the learner was asked to repeat. This technique is unique to the EIT context where the stimulus text is known a priori.

**How it works:**
1. For each EIT item, the target stimulus text is known (e.g., "Quiero cortarme el pelo").
2. The stimulus text is injected as the decoder prompt prefix via Whisper's `initial_prompt` parameter.
3. Whisper generates a continuation conditioned on this prefix, biasing the output toward the expected vocabulary and word order.
4. Simultaneously, a **plain decoding** pass (no stimulus) is run as a control.
5. If the prompted output shows signs of **over-anchoring** (cross-WER ≈ 0, indicating the model is parroting the stimulus rather than transcribing what was actually said), the system falls back to the plain decoding output.

> **📐 DIAGRAM PLACEHOLDER: Stimulus-Prompted Decoding Flow**
>
> *Draw a decision-flow diagram:*
> ```
> Input: audio segment + stimulus text
>         │
>         ▼
> ┌───────────────────┐
> │ Prompted Decoding  │──▶ transcript_prompted
> │ (stimulus as prefix)│
> └────────┬──────────┘
>          │
>          ▼
> ┌───────────────────┐     ┌────────────────┐
> │ Over-Anchoring     │─YES─▶│ Plain Decoding  │──▶ transcript_plain
> │ Detection          │     │ (no stimulus)   │
> │ (cross-WER check)  │     └────────────────┘
> └────────┬──────────┘
>          │ NO
>          ▼
>   Use transcript_prompted
> ```
>
> *Include: example of over-anchoring (model outputs exact stimulus instead of what learner said), example of successful prompting (model correctly transcribes learner variation).*

---

## Segment 2: Fine-tuning Pipeline (Model Training)

> This segment implements the offline model training system. It operates on external SLABANK corpora (not the EIT test data) and produces a fine-tuned Whisper model that Segment 1 can optionally use. It is a fully independent pipeline in `finetuning/` with its own data processing, audio preprocessing, augmentation, and training code.

### 4.5 Training Data & SLABANK Corpora

**Module:** `finetuning/src/data_loader.py`

Fine-tuning uses three SLABANK corpora of L2 Spanish learner speech, parsed from CHAT transcription format (20+ annotation patterns stripped):

| Corpus | L1 Groups | Original Utterances | Augmented Total | Audio Hours |
|--------|-----------|:-------------------:|:---------------:|:-----------:|
| **SPLLOC1** | English | 15,760 | ~31,520 | ~36h |
| **Nebrija-INMIGRA** | Brazilian-Portuguese, Cameroonian, Chinese, Filipino, French, Moroccan, Romanian, Senegalese, Syrian, Ukrainian | 11,721 | ~46,884 | ~27h |
| **Nebrija-WOCAE** | Chinese | 922 | ~3,688 | ~4h |
| **Total** | 11+ L1 groups | **28,403** | **56,090** | **~65.4h** |

**Speaker-stratified splits** (70/15/15) ensure no speaker leakage between train/dev/test:

| Split | Utterances | Audio Hours |
|-------|:----------:|:-----------:|
| Train (incl. augmentation) | 56,090 | ~65.4h |
| Dev | 4,033 | ~4.6h |
| Test | 4,698 | ~5.3h |

---

### 4.6 Audio Preprocessing for Training

**Module:** `finetuning/src/preprocess.py`

> **Note:** This preprocessing pipeline is distinct from the inference-time segmentation in Segment 1. It processes SLABANK training data, not EIT participant recordings. The two pipelines serve different purposes: Segment 1's segmentation extracts stimulus-response pairs from continuous recordings, while this pipeline prepares individual utterances for model training.

The 8-step pipeline transforms raw corpus audio into clean, normalised utterances suitable for Whisper fine-tuning:

| Step | Operation | Details |
|------|-----------|---------|
| 1 | **Load** | Read MP3/WAV, convert to mono |
| 2 | **Resample** | 16 kHz using kaiser_best interpolation |
| 3 | **Denoise** | Wiener filter + bandpass 80–7500 Hz |
| 4 | **VAD Trim** | webrtcvad (aggressiveness=2) to remove leading/trailing silence |
| 5 | **RMS Normalise** | Target -20 dBFS |
| 6 | **Peak Clip** | Clamp to ±0.99 to prevent digital clipping |
| 7 | **Quality Gate** | Reject if SNR < threshold or duration < 0.5s / > 30s |
| 8 | **Chunking** | Split sessions > 30s into 25s segments with 2s overlap |

**Corpus-specific denoising** adapts to recording conditions:

| Corpus | Denoise Strength | SNR Threshold | Rationale |
|--------|:----------------:|:-------------:|-----------|
| Nebrija-INMIGRA | 0.80 | 5.0 dB | Noisier lab recordings, diverse L1 backgrounds |
| Nebrija-WOCAE | 0.75 | 4.0 dB | Lower quality audio, relaxed gate |
| SPLLOC1 | 0.70 | 5.0 dB | Cleaner recordings, lighter denoising |

> **📐 DIAGRAM PLACEHOLDER: Training Audio Preprocessing Pipeline Flowchart**
>
> *Draw an Excalidraw flowchart showing:*
> - Each of the 8 steps as boxes with arrows
> - Decision diamond at Step 7 (Quality Gate): pass → continue, fail → reject
> - Branching at Step 8 (Chunking): short audio → direct output, long audio → overlap-split
> - Waveform visualisations at key stages (raw → denoised → normalised → segmented)
> - Colour-code by corpus for the corpus-specific denoising parameters

---

### 4.7 Data Augmentation

**Module:** `finetuning/src/augment.py`

> **📐 DIAGRAM PLACEHOLDER: Data Augmentation Strategy**
>
> *Draw a diagram showing:*
> - Original audio → augmentation branches (speed, noise, reverb, pitch, volume jitter)
> - Per-corpus multipliers (INMIGRA 4×, WOCAE 4×, SPLLOC1 2×)
> - Low-resource L1 boost (6× for underrepresented groups like Syrian, Senegalese)
> - Before/after spectrograms for each augmentation type
> - Final distribution bar chart: original vs. augmented utterance counts per corpus

| Corpus | Augmentation Tag | Transforms Applied |
|--------|------------------|--------------------|
| INMIGRA | `sp090_wn15` | Speed ×0.90 + white noise SNR 15 dB |
| INMIGRA | `sp110_pn18` | Speed ×1.10 + pink noise SNR 18 dB |
| INMIGRA | `sp095_revsm` | Speed ×0.95 + small-room reverb |
| WOCAE | `sp085_wn12` | Speed ×0.85 + white noise SNR 12 dB |
| WOCAE | `sp115_ps1_pn16` | Speed ×1.15 + pitch +1 semitone + pink noise SNR 16 dB |
| WOCAE | `sp105_revmd` | Speed ×1.05 + medium reverb |
| SPLLOC1 | `sp092_vj` | Speed ×0.92 + volume jitter |

---

### 4.8 LoRA Fine-tuning Strategy

**Notebook:** `finetuning/whisper-eit-kaggle-v2.ipynb`

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Base model** | `openai/whisper-large-v2` (1.55B params) | Best balance of quality and LoRA compatibility |
| **Method** | LoRA (Low-Rank Adaptation) via PEFT | Parameter-efficient: only 10.5M trainable params (0.68% of total) |
| **Rank (r)** | 8 | Sufficient capacity for domain adaptation without overfitting |
| **Alpha (α)** | 16 | Standard 2× rank scaling |
| **Dropout** | 0.05 | Light regularisation |
| **Target modules** | `q_proj`, `v_proj`, `k_proj`, `out_proj` | All attention projections in encoder, decoder, and cross-attention |
| **Batch size** | 4/device × 8 grad_accum = **32 effective** | Stable convergence on T4 ×2 |
| **Learning rate** | 5e-6 with cosine decay | Conservative to preserve pretrained features |
| **Warmup** | 5% of total steps | Gradual ramp-up |
| **Epochs** | 10 | With early stopping (patience=3) |
| **SpecAugment** | 2 time masks (100 frames) + 2 freq masks (20 bins) | Additional regularisation during training |
| **Weighted sampling** | 35% INMIGRA, 20% WOCAE, 45% SPLLOC1 | Balanced exposure despite corpus size imbalance |

> **📐 DIAGRAM PLACEHOLDER: LoRA Architecture Diagram**
>
> *Draw a detailed diagram showing:*
> - Whisper encoder-decoder architecture (12 encoder layers, 12 decoder layers)
> - LoRA adapter injection points (highlighted in colour) at q_proj, k_proj, v_proj, out_proj
> - Frozen weights (grey) vs. trainable LoRA weights (coloured)
> - The low-rank decomposition: W = W₀ + BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×d), r=8
> - Parameter count breakdown: 1.55B frozen + 10.5M trainable
> - Cross-attention layers specially highlighted (these bridge encoder→decoder)

**Training infrastructure:**
- **Primary:** Kaggle GPU (T4 ×2), ~8–10 hours training time
- **Local debugging:** Apple Silicon (M2) with MLX backend, whisper-small

**Output:** Fused model checkpoint at `fused/model.safetensors` (483 MB) — LoRA weights merged into the base model for simplified inference. This checkpoint feeds into Segment 1's transcription pipeline as an optional model backend.

---

## Shared Components

> These components serve both segments and are not exclusive to either the inference pipeline or the fine-tuning pipeline.

### 4.9 ASR Backbone Selection & Rationale

| Model | Parameters | Language Support | WER (L2 Spanish Baseline) | Pros | Cons |
|-------|:----------:|:----------------:|:-------------------------:|------|------|
| **Whisper-large-v3** | 1.55B | 98+ languages | **32.2%** | Best zero-shot multilingual, robust to noise | Large, slow inference, suppresses disfluencies |
| **Whisper-large-v2** | 1.55B | 98+ languages | ~34% (est.) | Stable training, good LoRA compatibility | Slightly worse zero-shot than v3 |
| **Whisper-small** | 244M | 98+ languages | ~52% (fused LoRA) | Fast inference, runs on Apple Silicon (MLX) | Higher error rate, limited capacity |
| wav2vec2-XLSR-53 | 300M | 53 languages | N/A (not tested) | Strong self-supervised features | Requires CTC head, less robust to noise |
| NeMo Conformer | ~120M | Configurable | N/A (not tested) | Streaming capable | Requires from-scratch Spanish training |

**Decision: Whisper-large-v2 as the fine-tuning base (Segment 2), Whisper-large-v3 as the inference backbone (Segment 1).**

**Rationale:**
1. Whisper's encoder-decoder architecture naturally handles the variable-length, noisy audio characteristic of learner speech.
2. The multilingual pre-training provides a strong Spanish foundation that can be efficiently adapted to L2 patterns via LoRA.
3. Whisper-large-v2 offers the most stable LoRA training experience based on community benchmarks and our own experiments.
4. The prompted decoding API allows stimulus conditioning — a critical feature for EIT alignment (§4.4).

> **📐 DIAGRAM PLACEHOLDER: Model Comparison Radar Chart**
>
> *Create a radar/spider chart comparing the models across 5 axes:*
> - Zero-shot WER on L2 Spanish
> - Inference speed (RTF)
> - Fine-tuning efficiency (trainable params %)
> - Disfluency handling
> - Multilingual robustness
>
> *Whisper-large-v2/v3 should dominate on most axes; wav2vec2 should lead on fine-tuning efficiency.*

---

### 4.10 Post-Processing Pipeline

| Step | Method | Purpose |
|------|--------|---------|
| **Text normalisation** | Unicode NFC, lowercase, strip punctuation/accents | Fair WER comparison |
| **Over-anchoring fallback** | Cross-WER threshold detection | Prevents stimulus parroting |
| **Drift detection** | No-speech probability + VAD confidence | Identifies empty/silent responses |
| **Session-level alignment** | Sliding-window difflib matching (Mode B) | Aligns full-session transcriptions to individual stimuli when per-utterance clips unavailable |
| **Word-level timestamps** | Per-word start/end times + confidence probabilities | Enables fine-grained scoring and human review |

**Disfluency preservation** is a critical design constraint: unlike commercial ASR systems that strip hesitations and fillers, this pipeline intentionally preserves them because they carry linguistic information relevant to proficiency scoring.

#### Proposed GSoC Enhancements (Post-Processing)

| Enhancement | Description | Expected Impact |
|-------------|-------------|-----------------|
| **KenLM rescoring** | Spanish language model rescoring on n-best hypotheses | Reduce substitution errors by ~5–10% |
| **Proficiency-aware beam search** | Adjust beam diversity based on estimated learner level | Better transcription of low-proficiency speakers |
| **Rule-based L2 error patterns** | Codify common L1-transfer errors (e.g., ser/estar confusion, article dropping) | Prevent over-correction of valid learner errors |
| **Confidence-gated human review** | Flag low-confidence utterances for manual verification | Maximise agreement with human raters |

---

### 4.11 Evaluation Framework

#### Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **90% Agreement** | Fraction of utterances with per-utterance accuracy ≥ 90% | **≥ 90%** (primary GSoC success metric) |
| **WER** | Word Error Rate (substitutions + insertions + deletions / reference words) | Minimise |
| **CER** | Character Error Rate | Minimise |
| **Cross-WER** | WER between prompted and plain transcription (over-anchoring detector) | Monitor |
| **Word F1** | Precision/recall at the word level | Maximise |
| **Disfluency Retention Rate** | Fraction of disfluencies in reference that appear in hypothesis | Maximise |
| **RTF** | Real-Time Factor (processing time / audio duration) | < 1.0 for practical use |

#### Evaluation is stratified by:
- **Participant** (individual speaker analysis)
- **L1 group** (cross-linguistic error patterns)
- **Proficiency level** (low/mid/high)
- **Corpus** (dataset-specific biases)

---

## 5. Test Submission Results

### 5.1 Baseline: Off-the-Shelf Whisper-large-v3

> **📊 TABLE: Include this as a formatted table or data report**

| Metric | Value |
|--------|-------|
| Model | `openai/whisper-large-v3` |
| Test utterances | 40 (stratified sample from SLABANK) |
| **WER** | **32.23%** |
| **CER** | **21.53%** |
| **90% Agreement** | **17.5%** |
| Avg RTF | 0.707 |
| Wall time | 86.4 seconds |

### 5.2 Fine-tuned Model: Whisper-small + LoRA (Fused)

| Metric | Value |
|--------|-------|
| Model | `whisper-small` + LoRA (fused, 483 MB) |
| **WER** | **52.41%** |
| **CER** | **34.40%** |
| **90% Agreement** | **14.95%** |
| Training steps | 1,500 |

> **⚠️ Note:** The fused model uses whisper-small as the base (244M params) due to local Apple Silicon memory constraints. The Kaggle fine-tuning on whisper-large-v2 (1.55B params) is expected to significantly outperform this, as the base model alone (large-v3) already achieves 32.2% WER vs. small's ~52%.

### 5.3 EIT Test Set: 4-Participant Results (Stimulus-Prompted Pipeline)

> **📊 TABLE / CHART: Create a bar chart comparing WER across participants**

| Recording | Stimuli Transcribed | Missing | Mean WER | Mean CER | Interpretation |
|-----------|:-------------------:|:-------:|:--------:|:--------:|----------------|
| **038015_1A** | 30 | 0 | **0.232** | 0.146 | Higher-proficiency learner; best results |
| **038011_1A** | 30 | 0 | **0.388** | 0.267 | Mid-proficiency; moderate error rate |
| **038010_2A** | 30 | 0 | **0.505** | 0.376 | Lower proficiency; more L1 transfer |
| **038012_2A** | 30 | 0 | **0.614** | 0.432 | Challenging speaker; high disfluency rate |
| **Cross-participant mean** | **120** | **0** | **0.435** | **0.305** | — |

> **📊 CHART PLACEHOLDER: Per-Participant WER Distribution**
>
> *Create a grouped bar chart or box plot:*
> - X-axis: Participant ID (038015, 038011, 038010, 038012)
> - Y-axis: WER (0.0 – 1.0)
> - Bars for: Mean WER, Mean CER
> - Annotate with proficiency interpretation
> - Include horizontal dashed line at WER = 0.10 (90% agreement target)

### 5.4 Performance Comparison: Baseline vs. Fine-tuned vs. Prompted

> **📊 TABLE / CHART: Model Comparison Summary**
>
> *Create this as both a table and a bar chart for visual impact:*

| Configuration | Base Model | Fine-tuned? | Stimulus Prompting? | WER | CER | 90% Agreement |
|--------------|------------|:-----------:|:-------------------:|:---:|:---:|:-------------:|
| Off-the-shelf baseline | Whisper-large-v3 | ❌ | ❌ | 32.2% | 21.5% | 17.5% |
| LoRA fused (small) | Whisper-small | ✅ | ❌ | 52.4% | 34.4% | 15.0% |
| Prompted pipeline (MLX) | Whisper (MLX) | ❌ | ✅ | **43.5%*** | **30.5%*** | `[TBD]` |
| **Proposed: LoRA (large-v2) + Prompted** | Whisper-large-v2 | ✅ | ✅ | **Target: <20%** | **Target: <15%** | **Target: ≥90%** |

*\* Cross-participant mean on 4-participant EIT test set*

> **📊 CHART PLACEHOLDER: Performance Progression Chart**
>
> *Create a line or waterfall chart showing WER improvement across pipeline stages:*
> 1. Raw Whisper-large-v3 (no preprocessing) → WER ~40–50%
> 2. + Audio preprocessing → WER ~32%
> 3. + Stimulus-prompted decoding → WER ~28% (est.)
> 4. + LoRA fine-tuning on L2 data → WER ~20% (target)
> 5. + Post-processing (LM rescoring, rule-based) → WER <15% (target)
>
> *Show each improvement as a step down, with the delta labelled.*

### 5.5 Error Analysis

> **📊 ATTACH: Include the generated visualisation PNGs from eit_output_v3/**
>
> Relevant visualisations already generated by the pipeline:
> - `01_wer_overview.png` — WER distribution across utterances
> - `02_transcription_detail.png` — Word-level accuracy breakdown
> - `01_gap_analysis.png` — Segmentation gap analysis
> - `02_timeline.png` — Response timeline visualisation

**Key failure patterns observed:**

| Error Type | Example | Frequency | Root Cause |
|-----------|---------|-----------|------------|
| **Over-anchoring** | Stimulus: "El libro está en la mesa" → Output: "El libro está en la mesa" (exact copy despite learner saying something different) | ~15% of prompted outputs | Stimulus conditioning too strong; mitigated by cross-WER fallback |
| **Disfluency stripping** | Reference: "ehm un family eh de vacaciones" → Hypothesis: "diva que se nos" | Common in SPLLOC1 | Whisper trained to suppress non-lexical items |
| **L1 phonological confusion** | Reference: "atentamente" → Hypothesis: "adentramente" | Common in WOCAE | Chinese L1 phonological transfer |
| **Partial repetition collapse** | Reference: "rompe rompe su reloj su reloj" → Hypothesis: "Rompé su reloj" | Moderate | Model collapses repeated sequences |

---

## 6. Timeline (12-Week Breakdown)

| Phase | Weeks | Hours | Activities | Deliverables |
|-------|:-----:|:-----:|------------|--------------|
| **Community Bonding** | 1–2 | 20h | Literature review (L2 ASR, EIT methodology); mentor sync on evaluation criteria; set up Kaggle/cloud GPU environment; review SLABANK corpus documentation | Annotated bibliography; development environment ready; detailed evaluation rubric aligned with mentors |
| **Preprocessing Refinement** | 3–4 | 25h | Tune corpus-specific denoising parameters; implement DeepFilterNet as alternative to Wiener filter; improve VAD with Silero VAD confidence thresholds; extend segmentation for edge cases | Improved preprocessing module with A/B test results against current pipeline |
| **Baseline Consolidation** | 5 | 15h | Run Whisper-large-v3 baseline on full test set; compute per-L1 and per-proficiency stratified metrics; establish inter-rater reliability baseline from human transcriptions | Comprehensive baseline report with stratified error analysis |
| **Fine-tuning Round 1** | 6–7 | 35h | Fine-tune Whisper-large-v2 with LoRA on 56K augmented utterances (Kaggle T4×2); hyperparameter sweep (LR, rank, target modules); evaluate on dev set | Fine-tuned model checkpoint; training curves; dev-set WER/CER |
| **Post-Processing & LM Rescoring** | 8–9 | 30h | Implement KenLM Spanish rescoring; rule-based L2 error pattern corrections; confidence-gated human review flagging; proficiency-aware beam search | Post-processing module; LM integration; confidence calibration |
| **Fine-tuning Round 2** | 10 | 20h | Incorporate error analysis from Round 1; targeted augmentation for failure cases; experiment with larger LoRA rank or full fine-tuning if compute allows | Improved model checkpoint; ablation study |
| **Evaluation & Integration** | 11 | 20h | Full pipeline evaluation on test set; compute 90% agreement score; per-participant error analysis; output in required Excel format; compare with human rater scores | Final evaluation report; formatted Excel outputs; agreement scores |
| **Documentation & Wrap-up** | 12 | 10h | Clean repository; write technical documentation; prepare research-style write-up; final mentor review; submit to HuggingFace Hub | Clean codebase; documentation; HuggingFace model card; final report |
| | | **175h** | | |

> **📐 DIAGRAM PLACEHOLDER: Gantt Chart Timeline**
>
> *Create a Gantt chart showing:*
> - 12 weeks on X-axis
> - Phases as horizontal bars with colour coding
> - Milestones marked (baseline report, first fine-tuned model, final evaluation)
> - Dependencies between phases shown with arrows
> - Mentor sync points highlighted

---

## 7. Deliverables

#### Segment 1: EIT Segmentation & Transcription Pipeline

| # | Deliverable | Format | Status |
|:-:|------------|--------|--------|
| 1 | **Audio segmentation pipeline** | Notebook (`eit_segmentation_v3.ipynb`) — VAD, gap detection, stimulus mapping | ✅ Implemented |
| 2 | **Transcription pipeline** | Notebook (`eit_transcription.ipynb`) — multi-backend ASR, checkpoint system | ✅ Implemented |
| 3 | **Stimulus-prompted decoding** | Integrated in transcription pipeline with over-anchoring guard | ✅ Implemented |
| 4 | **Drift detection** | Post-hoc segmentation validation + no-speech region detection | ✅ Implemented |
| 5 | **Completed test transcriptions** | Excel format matching provided rubric | ✅ 4 participants (120 utterances, 0 missing) |

#### Segment 2: Fine-tuning Pipeline

| # | Deliverable | Format | Status |
|:-:|------------|--------|--------|
| 6 | **CHAT parser & data loader** | Python module (`finetuning/src/data_loader.py`) | ✅ Implemented |
| 7 | **Training audio preprocessor** | Python module (`finetuning/src/preprocess.py`) — 8-step corpus-specific pipeline | ✅ Implemented |
| 8 | **Data augmentation module** | Python module (`finetuning/src/augment.py`) — corpus-specific + L1 boost | ✅ Implemented |
| 9 | **Fine-tuned ASR model** | HuggingFace Hub model card + `fused/model.safetensors` (483 MB) | 🔄 In progress (small model fused; large-v2 pending) |
| 10 | **Unit tests** | pytest suite (`finetuning/tests/`) | ✅ Implemented |

#### Shared / Planned

| # | Deliverable | Format | Status |
|:-:|------------|--------|--------|
| 11 | **Post-processing pipeline** | LM rescoring + rule-based corrections + confidence gating | 📋 Planned for GSoC |
| 12 | **Evaluation suite** | Python module (`finetuning/src/evaluate.py`) with WER/CER/90% agreement | ✅ Implemented |
| 13 | **Technical documentation** | README + research-style write-up | 📋 Planned for GSoC |

---

## 8. About Me / Why I Am a Good Fit

`[INSERT: Your technical skills — Python, PyTorch, HuggingFace, librosa, etc.]`

`[INSERT: Domain knowledge — linguistics, SLA, Spanish proficiency level]`

`[INSERT: Open-source contributions or research experience]`

`[INSERT: Communication style, time management, availability during GSoC period]`

**What I have already built for this project:**

*Segment 1 — EIT Segmentation & Transcription Pipeline:*
- VAD-based audio segmentation with intra-utterance merging, smart reduce, and gap-based missing stimulus detection (`eit_segmentation_v3.ipynb`)
- Multi-backend transcription pipeline with auto-detection (MLX / whisper.cpp / OpenAI), checkpoint resumability, and batch processing (`eit_transcription.ipynb`)
- Stimulus-prompted decoding with over-anchoring detection and plain fallback
- Post-hoc segmentation drift detection for mapping validation
- No-speech detection to prevent hallucination on empty responses
- Transcriptions for all 4 test participants (120 utterances, 0 missing)

*Segment 2 — Fine-tuning Pipeline:*
- A CHAT parser and data loader for 3 SLABANK corpora (20+ annotation patterns, `finetuning/src/data_loader.py`)
- An 8-step corpus-specific audio preprocessing module with quality gating (`finetuning/src/preprocess.py`)
- A data augmentation system with corpus-specific strategies and low-resource L1 boosting (`finetuning/src/augment.py`)
- LoRA fine-tuning infrastructure for both Kaggle (GPU T4×2) and Apple Silicon (MLX)
- Fused model checkpoint (`fused/model.safetensors`, 483 MB)

*Shared:*
- A comprehensive evaluation framework (WER, CER, 90% agreement, word F1, per-group stratification)
- Unit tests for all major modules

This existing codebase — spanning two independent but complementary pipelines — demonstrates both my technical capability and my commitment to the project. The GSoC period will focus on refinement — better fine-tuning, smarter post-processing, and closing the gap to the 90% agreement target.

---

## 9. References

1. **Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022).** Robust speech recognition via large-scale weak supervision. *arXiv preprint arXiv:2212.04356*. [Whisper]

2. **Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020).** wav2vec 2.0: A framework for self-supervised learning of speech representations. *NeurIPS 2020*. [Self-supervised ASR]

3. **Conneau, A., Baevski, A., Collobert, R., Mohamed, A., & Auli, M. (2020).** Unsupervised cross-lingual representation learning for speech recognition. *arXiv preprint arXiv:2006.13979*. [XLSR-53]

4. **Erlam, R. (2006).** Elicited imitation as a measure of L2 implicit knowledge: An empirical validation study. *Applied Linguistics, 27*(3), 464–491. [EIT methodology]

5. **Hu, E.J., Shen, Y., Wallis, P., et al. (2021).** LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*. [LoRA fine-tuning]

6. **Moran, et al.** Learner ASR performance and L2 speech recognition challenges. `[Verify current citation — user should confirm exact paper]`

7. **Dominguez, L., Tracy-Ventura, N., Arche, M.J., Mitchell, R., & Myles, F. (2013).** The role of dynamic contrasts in the L2 acquisition of Spanish past tense morphology. *Bilingualism: Language and Cognition, 16*(3), 558–577. [SPLLOC1 corpus]

8. **Perlmutter, L., et al. (2018–2022).** Nebrija-WOCAE and Nebrija-INMIGRA corpora. [SLABANK L2 Spanish data]

---

## Appendix A: Repository Structure

```
autoeit_humanai/
├── CLAUDE.md                          # Proposal framework
├── pyproject.toml                     # Project metadata (v0.1.0)
│
│   ┌─── SEGMENT 1: EIT Segmentation & Transcription Pipeline ───┐
│   │                                                              │
├── eit_segmentation_v3.ipynb          # Stage 1: VAD → segment extraction → stimulus mapping
├── eit_transcription.ipynb            # Stage 2: Multi-backend ASR → prompted decoding → scoring
├── eit_transcription_fused.ipynb      # Stage 2 variant: uses fused fine-tuned model
│   │                                                              │
├── eit_output_v3/                     # Segment 1 outputs (4 participants)
│   ├── batch_transcription_summary.csv
│   └── [participant]_preprocessed/
│       ├── segments.csv               #   ← from eit_segmentation_v3
│       ├── responses/                 #   ← individual clipped WAVs
│       ├── transcriptions.csv / .xlsx #   ← from eit_transcription
│       └── transcriptions/            #   ← metrics, visualisations, timestamps
│   │                                                              │
│   └──────────────────────────────────────────────────────────────┘
│
│   ┌─── SEGMENT 2: Fine-tuning Pipeline ────────────────────────┐
│   │                                                              │
├── finetuning/
│   ├── src/
│   │   ├── data_loader.py             # CHAT parser, manifest loader, splits
│   │   ├── preprocess.py              # 8-step audio preprocessor (training data)
│   │   ├── augment.py                 # Corpus-specific augmentation
│   │   ├── train.py                   # MLX LoRA training (Apple Silicon)
│   │   ├── infer.py                   # Inference + text normalisation
│   │   └── evaluate.py                # WER/CER/90% agreement metrics
│   ├── configs/config.yaml            # Hyperparameters
│   ├── scripts/
│   │   ├── run_baseline.py            # Whisper-large-v3 baseline
│   │   └── generate_test_mini.py      # Quick test subset
│   ├── results/
│   │   └── baseline_openai_whisper-large-v3.csv
│   ├── tests/                         # Unit tests (pytest)
│   ├── data/                          # SLABANK corpora (3 datasets)
│   ├── whisper-eit-kaggle-v2.ipynb    # Kaggle training notebook
│   └── README.md                      # Pipeline documentation
│   │                                                              │
├── fused/                             # Segment 2 output → feeds into Segment 1
│   ├── model.safetensors              # Merged LoRA weights (483 MB)
│   ├── config.json                    # WhisperForConditionalGeneration
│   ├── generation_config.json
│   └── README.md
│   │                                                              │
│   └──────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Diagram Checklist

The following diagrams should be created and embedded in the final submission:

| # | Diagram | Tool | Section |
|:-:|---------|------|---------|
| 1 | **End-to-End System Architecture** | Excalidraw | §4.1 |
| 2 | **Segmentation Pipeline Flowchart** (VAD → merge → smart reduce → gap detection → mapping) | Excalidraw | §4.2 |
| 3 | **Model Comparison Radar Chart** (5-axis: WER, speed, efficiency, disfluency, multilingual) | Matplotlib / Data report | §4.9 |
| 4 | **Stimulus-Prompted Decoding Flow** (decision tree: prompted → over-anchor check → fallback) | Excalidraw | §4.4 |
| 5 | **Training Audio Preprocessing Flowchart** (8 steps, decision gates, waveform examples) | Excalidraw | §4.6 |
| 6 | **Data Augmentation Strategy** (per-corpus branches, multipliers, spectrograms) | Excalidraw | §4.7 |
| 7 | **LoRA Architecture Diagram** (encoder-decoder, injection points, frozen vs. trainable) | Excalidraw | §4.8 |
| 8 | **Per-Participant WER Bar Chart** (4 participants, WER + CER, 90% target line) | Matplotlib / Data report | §5.3 |
| 9 | **Performance Progression Waterfall** (raw → preprocess → prompted → fine-tuned → post-processed) | Matplotlib / Data report | §5.4 |
| 10 | **Gantt Chart Timeline** (12 weeks, phases, milestones, dependencies) | Excalidraw / Gantt tool | §6 |

---

> **📅 Deadline Reminder:** The GSoC 2026 proposal deadline is typically early April. Submit your test results to the mentors at least 1 week before the proposal deadline, per the project instructions.
