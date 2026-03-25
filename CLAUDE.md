# CLAUDE.md — AutoEIT GSoC 2026 Proposal Generator

## Purpose
This file gives Claude all the context it needs to help draft, refine, and critique
a Google Summer of Code (GSoC) 2026 proposal for the **HumanAI / AutoEIT** project:
**"Audio-to-text transcription for second/additional language learner data"**.

When the user asks you to "write the proposal", "improve section X", "make it more
technical", or anything related to this proposal, use everything in this file as your
ground truth and follow the instructions in the TASK section at the bottom.

---

## 1. Project Background

### 1.1 What is the AutoEIT project?
AutoEIT is a research tool being developed under the HumanAI umbrella (CERN-affiliated
programme) in collaboration with:
- **Northern Illinois University (NIU)** — Dr. Mandy Faretta-Stutenberg
- **University of Alabama** — Dr. Xabier Granja

Its goal is to automate the end-to-end pipeline for the **Spanish Elicited Imitation
Task (EIT)**: from raw learner audio → transcription → proficiency scoring.

### 1.2 What is an Elicited Imitation Task (EIT)?
- A sentence-repetition task used in second language acquisition (SLA) research.
- Participants hear a Spanish sentence and immediately repeat it from memory.
- Their production reveals gaps in grammatical knowledge, phonological accuracy, and
  overall proficiency.
- A single session yields ~30 sentences per participant.
- Scoring is currently done manually by trained human raters — extremely time-intensive
  at scale.

### 1.3 Why is ASR hard here?
Standard automatic speech recognition (ASR) systems (Whisper, Google STT, Azure, etc.)
are trained predominantly on native or near-native speech. Learner language breaks these
assumptions in several ways:

| Challenge | Description |
|-----------|-------------|
| L1 transfer | Phonemes from the learner's native language bleed into Spanish production |
| Variable proficiency | Beginners through advanced learners in the same dataset |
| Disfluencies | Hesitations, false starts, self-corrections, fillers |
| Partial repetitions | Learners may reproduce only part of the target sentence |
| Codeswitching | Occasional English (or other L1) words inserted into Spanish output |
| Audio quality | Lab recordings but with varying microphone setups and background noise |

### 1.4 Dataset (Test Files)
The evaluation dataset provided to applicants contains:
- Audio files for **4 participants**, each completing the Spanish EIT (~30 sentences each)
- An Excel file (`AutoEIT Sample Audio for Transcribing.xlsx`) with target sentences,
  human transcriptions, and the EIT scoring rubric
- Data is organised with one tab per participant

### 1.5 Project Scope (GSoC 2026)
- **Track:** Audio-to-text transcription (Test I)
- **Total hours:** 175 hours
- **Difficulty:** Medium
- **Required skills:** Python, PyTorch or TensorFlow, ML experience
- **Success metric:** ≥90% agreement with experienced human transcribers

---

## 2. Codebase Context

The applicant has an existing codebase in their GitHub repository. When the user shares
details about the repo (structure, scripts, models used, notebooks), incorporate those
specifics into every proposal section that discusses implementation.

Key things to note when reading the repo:
- Which ASR backbone is used (Whisper, wav2vec2, NeMo, etc.)?
- Is there a fine-tuning script? What dataset was used?
- What preprocessing steps are implemented (denoising, VAD, segmentation)?
- What post-processing is implemented (language model rescoring, error correction)?
- What evaluation metrics are computed (WER, CER, human agreement %)?
- What is the output format (Excel, JSON, plain text)?

If the user pastes file contents or a directory tree, update your understanding of the
codebase and reflect it accurately in the proposal.

---

## 3. Proposal Structure

A strong GSoC proposal for this project must contain **all** of the following sections,
in roughly this order. Adjust length to context but never omit a section.

### 3.1 Personal Information & Synopsis (≤200 words)
- Name, university, degree programme, country, timezone
- One-paragraph executive summary of the proposal
- Link to GitHub repo with test submission

### 3.2 Motivation & Background (≤300 words)
- Why this specific project? (genuine interest in SLA/NLP intersection)
- Relevant coursework, research experience, or personal background
- Prior exposure to multilingual NLP or speech processing
- Familiarity with linguistic concepts (L2 acquisition, phonological transfer, EIT)

### 3.3 Understanding of the Problem (≤400 words)
- Explain the EIT, why manual transcription is the bottleneck, and what makes learner
  speech uniquely difficult for ASR
- Demonstrate awareness of relevant literature (e.g., Moran et al. on learner ASR,
  whisper-based L2 studies, wav2vec2 fine-tuning for non-native speech)
- Identify the specific failure modes of off-the-shelf ASR on this data

### 3.4 Technical Approach (the core section — ≥600 words)

The project consists of **two distinct segments** that should be described separately:

**Segment 1 — EIT Segmentation & Transcription Pipeline** (`eit_segmentation_v3.ipynb`
+ `eit_transcription.ipynb`): the inference-time system that processes raw participant
audio into scored transcriptions. This is the baseline pipeline that runs per participant.

**Segment 2 — Fine-tuning Pipeline** (`finetuning/`): an independent offline system with
its own data sources, audio preprocessing, augmentation, and model training. This produces
the fine-tuned model that Segment 1 can optionally use.

Break the technical approach into clearly labelled sub-sections that respect this
two-segment structure:

---

#### Segment 1: EIT Segmentation & Transcription Pipeline

##### 3.4.1 Audio Segmentation (`eit_segmentation_v3.ipynb`)
- Voice Activity Detection (Silero VAD) configured for non-native speech
  (lower thresholds to catch quieter L2 productions)
- Intra-utterance merging: collapse segments separated by short gaps to handle
  mid-utterance pauses and hesitations common in learner speech
- Smart reduce algorithm: if VAD yields > 30 segments, iteratively merge closest
  pairs until reaching the target count
- Gap-based missing stimulus detection: identify skipped responses using inter-segment
  gap analysis (bimodal distribution of intra-response vs. inter-pair gaps)
- Sequential stimulus mapping: assign detected speech segments to the 30 stimulus
  slots in temporal order
- Output: `segments.csv` (per-segment metadata) + individual response WAV files

##### 3.4.2 Transcription Pipeline (`eit_transcription.ipynb`)
- Multi-backend ASR architecture: auto-detect platform and select fastest backend
  (MLX on Apple Silicon, whisper.cpp, or OpenAI Whisper as fallback)
- Checkpoint system: JSON-based incremental saves for resumability
- Stimulus-prompted decoding: two-pass approach — prompted (stimulus as decoder prefix)
  vs. plain, with cross-WER comparison
- Over-anchoring detection: if prompted output suspiciously matches stimulus exactly
  (cross-WER ≈ 0), fall back to plain decoding
- Drift detection: post-hoc sliding-window validation against neighbouring stimuli
  to catch off-by-one mapping errors from segmentation
- No-speech detection: silence probability thresholding to identify empty responses
- Text normalisation for fair WER comparison (Unicode NFC, lowercase, punctuation-agnostic)
- Output: `transcriptions.csv/xlsx`, word-level timestamps, WER/CER metrics,
  visualisations, checkpoint JSON

---

#### Segment 2: Fine-tuning Pipeline (`finetuning/`)

##### 3.4.3 Training Data & CHAT Parsing (`src/data_loader.py`)
- Dataset curation from SLABANK L2 Spanish corpora (Nebrija-INMIGRA, Nebrija-WOCAE,
  SPLLOC1) — parsing CHAT transcription format with 20+ annotation patterns
- Speaker-stratified train/dev/test splits (no speaker leakage)
- Any supplemental L2 corpora (TELL-ME, CASL, or synthetic augmentation via TTS)

##### 3.4.4 Audio Preprocessing for Training (`src/preprocess.py`)
- 8-step pipeline: load → resample → denoise+bandpass → VAD trim → RMS normalise
  → peak clip → quality gate → chunking
- Corpus-specific denoising parameters (different SNR thresholds per corpus)
- This is distinct from the inference-time segmentation in Segment 1

##### 3.4.5 Data Augmentation (`src/augment.py`)
- Corpus-specific augmentation strategies (speed, noise, reverb, pitch, volume jitter)
- Low-resource L1 boosting (6× for underrepresented groups)
- Handling class imbalance across proficiency levels and L1 backgrounds

##### 3.4.6 Model Training (`whisper-eit-kaggle-v2.ipynb`)
- LoRA / parameter-efficient fine-tuning (r=8, α=16, targeting q/k/v/out_proj)
- Training infrastructure: Kaggle GPU (T4×2) for full training, Apple Silicon (MLX)
  for local debugging
- Hyperparameter choices (LR, batch size, scheduler, SpecAugment)
- Weighted sampling to balance corpus exposure
- Output: fused model checkpoint (`fused/model.safetensors`)

---

#### Shared Components

##### 3.4.7 ASR Backbone Selection & Rationale
- Compare options: OpenAI Whisper (large-v3 or turbo), Meta wav2vec2-large-xlsr-53,
  NVIDIA NeMo, SpeechBrain
- Justify the chosen model for learner Spanish specifically
- Discuss multilingual vs. Spanish-specific model trade-offs

##### 3.4.8 Post-Processing Pipeline
- Language-model rescoring (KenLM or neural LM constrained to Spanish)
- Rule-based corrections for predictable L2 errors (e.g., article dropping,
  ser/estar confusion transcribed incorrectly)
- Disfluency preservation: the pipeline must NOT silently remove hesitations,
  false starts, or fillers — these are linguistically meaningful
- Output formatting: sentence-level alignment to the target sentence list in the Excel

##### 3.4.9 Evaluation Framework
- Primary metric: percentage agreement with human transcripts (target ≥90%)
- Secondary metrics: WER, CER, disfluency retention rate
- Agreement computation: exact match vs. normalised match (punctuation-agnostic,
  case-insensitive, disfluency-aware)
- Inter-rater reliability baseline using the provided human transcriptions

### 3.5 Timeline (12-week breakdown)
Provide a week-by-week or phase-by-phase plan covering all 175 hours.
Example structure:

| Phase | Weeks | Hours | Deliverables |
|-------|-------|-------|--------------|
| Community bonding | 1–2 | ~20 | Repo setup, literature review, mentor sync |
| Preprocessing pipeline | 3–4 | ~30 | Noise reduction, VAD, segmentation scripts |
| Baseline ASR evaluation | 5 | ~15 | Off-the-shelf Whisper WER on test set |
| Fine-tuning | 6–8 | ~45 | Fine-tuned model checkpoint, training logs |
| Post-processing | 9–10 | ~30 | LM rescoring, rule-based corrections |
| Evaluation & refinement | 11 | ~20 | Agreement scores, error analysis |
| Documentation & wrap-up | 12 | ~15 | Final report, cleaned repo, mentor review |

### 3.6 Deliverables
List concrete, measurable outputs:
- Preprocessing module (Python package / script)
- Fine-tuned ASR model (with HuggingFace Hub link if public)
- Post-processing pipeline
- Evaluation scripts (agreement, WER, CER)
- Completed transcription of all test audio in the required Excel format
- Technical documentation and a brief research-style write-up

### 3.7 Test Submission Summary (≤300 words)
- Describe the approach taken for Test I
- Report the quantitative results (WER or agreement % on the 4 participants)
- Discuss challenges encountered and how they were addressed
- Link to the GitHub branch with all code and output files

### 3.8 About Me / Why I Am a Good Fit (≤300 words)
- Relevant technical skills (Python, PyTorch/TF, HuggingFace, librosa, etc.)
- Relevant domain knowledge (linguistics, SLA, Spanish proficiency)
- Open-source contributions or prior research projects
- Communication and time-management track record

---

## 4. Tone & Style Guidelines

- **Audience:** Two academic researchers (linguistics / SLA background) who are also
  technically literate but not necessarily ML engineers. Avoid excessive jargon without
  explanation; define all acronyms on first use.
- **Voice:** Confident but not arrogant. Show genuine curiosity about the linguistic
  problem, not just the ML challenge.
- **Precision:** Cite specific tools, model names, and papers. "I will use a
  speech recognition model" is weak; "I will fine-tune Whisper large-v3 using LoRA
  on augmented L2 Spanish data" is strong.
- **Honesty about uncertainty:** If there are open questions (e.g., "the best
  fine-tuning dataset is unclear and will be determined during community bonding"),
  state them openly and show a plan for resolving them.
- **Length:** A complete proposal should be 1,500–2,500 words excluding tables.

---

## 5. Common Mistakes to Avoid

1. **Ignoring disfluency preservation.** The rubric explicitly requires transcribing
   exact production, including disfluencies. Any approach that strips these out is wrong.
2. **Overclaiming accuracy.** Do not promise 95%+ agreement without data to back it up.
   The 90% target is already ambitious.
3. **Treating this as a standard ASR problem.** Emphasise the L2/learner-speech angle
   throughout. Generic ASR proposals will not stand out.
4. **Vague timeline.** Each week must have concrete tasks and hour estimates totalling
   exactly 175 hours.
5. **Forgetting the output format.** Results must be in the Excel format provided
   (or a clearly justified equivalent), not just a text dump.
6. **Not linking the test submission.** The mentors explicitly require a GitHub link
   with code and results before evaluating the proposal.

---

## 6. Key References to Cite (if applicable)

- Whisper (Radford et al., 2022) — OpenAI's multilingual ASR model
- wav2vec 2.0 (Baevski et al., 2020) — Self-supervised speech representation learning
- XLSR-53 (Conneau et al., 2020) — Cross-lingual speech representations
- Moran et al. (relevant L2 ASR work — user should verify current citations)
- Work on EIT methodology: Erlam (2006), Ortega et al. — foundational EIT papers
- HuggingFace `transformers` and `datasets` libraries for fine-tuning workflows

---

## 7. TASK INSTRUCTIONS FOR CLAUDE

When the user triggers proposal generation (e.g., "write the proposal", "draft section 3",
"improve the timeline"), follow these rules:

1. **Use all context above** as the single source of truth for project facts.
2. **Ask for missing information** before drafting if critical details are absent
   (e.g., the user's name, university, or test results). Collect them in one turn.
3. **Incorporate codebase specifics** whenever the user has shared repo details.
   Never invent code that doesn't exist in the repo.
4. **Generate the full proposal** when asked, section by section if long, following
   Section 3 structure exactly.
5. **Critique mode:** If the user pastes a draft and asks for feedback, evaluate it
   against the structure in Section 3 and the mistakes listed in Section 5.
6. **Formatting:** Output in clean Markdown suitable for direct pasting into the GSoC
   application portal or a Google Doc.
7. **Do not hallucinate test results.** If the user has not shared WER/agreement numbers,
   leave a `[INSERT RESULTS]` placeholder and remind them to fill it in.
8. **Deadline awareness:** GSoC 2026 proposal deadline is typically early April.
   Remind the user to submit their test results to the mentors at least 1 week before
   the proposal deadline, per the project instructions.

---

*Last updated: March 2026*