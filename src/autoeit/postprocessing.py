"""Post-processing module for ASR output.

Distinguishes between ASR errors (to fix) and participant errors (to preserve).

Rules in priority order:
  1. Accent/diacritical normalisation for valid Spanish words
  2. Hallucination detection (repeated phrases, silence-filled text)
  3. Disfluency formatting to match reference transcription conventions
  4. No-response detection
  5. Preserve all participant errors (gender, conjugation, word choice, etc.)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .config import PostProcessingConfig, TARGET_SENTENCES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spanish accent corrections — common Whisper mistakes
# ---------------------------------------------------------------------------

# Words where Whisper commonly drops or misplaces accents
ACCENT_CORRECTIONS: Dict[str, str] = {
    # Common words
    "esta": "está",   # but only as verb — context-dependent
    "asi": "así",
    "despues": "después",
    "tambien": "también",
    "aqui": "aquí",
    "ahi": "ahí",
    "alla": "allá",
    "dificil": "difícil",
    "facil": "fácil",
    "musica": "música",
    "pelicula": "película",
    "peliculas": "películas",
    "numero": "número",
    "telefono": "teléfono",
    "pajaro": "pájaro",
    "arbol": "árbol",
    "sabado": "sábado",
    "miercoles": "miércoles",
    "espanol": "español",
    "ingles": "inglés",
    "frances": "francés",
    "aleman": "alemán",
    "cafe": "café",
    "mama": "mamá",
    "papa": "papá",
    "salon": "salón",
    "ladron": "ladrón",
    "policia": "policía",
    "manana": "mañana",
    "nino": "niño",
    "nina": "niña",
    "ninos": "niños",
    "ano": "año",
    "anos": "años",
    "cuantos": "cuántos",
    "cuanto": "cuánto",
    # EIT-specific verbs and forms
    "gustaria": "gustaría",
    "seria": "sería",
    "serias": "serías",
    "tome": "tomé",
    "pedi": "pedí",
    "murio": "murió",
    "atrapo": "atrapó",
    "termino": "terminó",
    "empezara": "empezara",
    "bajara": "bajara",
    # Common past tenses
    "llego": "llegó",
    "paso": "pasó",
    "hablo": "habló",
    "ceno": "cenó",
}

# Words that should NOT be corrected (they are valid without accent)
NO_CORRECT_WORDS = {
    "el",     # article (vs "él" pronoun — keep as-is, context hard)
    "mi",     # possessive (vs "mí" pronoun)
    "tu",     # possessive (vs "tú" pronoun)
    "de",
    "se",
    "si",     # "if" vs "sí" (yes/self) — too ambiguous
    "como",   # "as/like" vs "cómo" "how" — context-dependent
    "que",    # relative vs interrogative
    "solo",   # RAE 2010 allows without accent
}


def postprocess_transcription(
    raw_text: str,
    stimulus: str,
    config: PostProcessingConfig,
    avg_log_prob: float = 0.0,
    no_speech_prob: float = 0.0,
    segment_energy: float = 0.0,
) -> Tuple[str, bool, str]:
    """Apply all post-processing rules to a single transcription.

    Parameters
    ----------
    raw_text : str
        Raw ASR output.
    stimulus : str
        The target sentence for this item.
    config : PostProcessingConfig
        Post-processing configuration.
    avg_log_prob : float
        ASR confidence (log probability).
    no_speech_prob : float
        Probability no speech was present.
    segment_energy : float
        RMS energy of the audio segment.

    Returns
    -------
    text : str
        Post-processed transcription.
    flagged : bool
        Whether this item needs manual review.
    flag_reason : str
        Reason for flagging (empty if not flagged).
    """
    text = raw_text.strip()
    flagged = False
    flag_reasons = []

    # ------------------------------------------------------------------
    # 1. No-response detection
    # ------------------------------------------------------------------
    if _is_no_response(text, segment_energy, config):
        return "[no response]", False, ""

    # ------------------------------------------------------------------
    # 2. Hallucination detection
    # ------------------------------------------------------------------
    if config.remove_hallucinations:
        text, was_hallucination = _remove_hallucinations(text, stimulus, config)
        if was_hallucination:
            flagged = True
            flag_reasons.append("possible_hallucination")

    # ------------------------------------------------------------------
    # 3. Accent/diacritical normalisation
    # ------------------------------------------------------------------
    if config.fix_accents:
        text = _fix_accents(text)

    # ------------------------------------------------------------------
    # 4. Disfluency formatting
    # ------------------------------------------------------------------
    if config.format_disfluencies:
        text = _format_disfluencies(text)

    # ------------------------------------------------------------------
    # 5. General cleanup
    # ------------------------------------------------------------------
    text = _cleanup(text)

    # ------------------------------------------------------------------
    # 6. Confidence-based flagging
    # ------------------------------------------------------------------
    if avg_log_prob < -1.5:
        flagged = True
        flag_reasons.append(f"low_confidence({avg_log_prob:.2f})")

    if no_speech_prob > 0.5:
        flagged = True
        flag_reasons.append(f"high_no_speech({no_speech_prob:.2f})")

    # Check if transcription is suspiciously close to stimulus
    similarity = _compute_similarity(text, stimulus)
    if similarity > 0.95:
        flagged = True
        flag_reasons.append(f"stimulus_echo(sim={similarity:.2f})")

    flag_reason = "; ".join(flag_reasons)
    return text, flagged, flag_reason


# ---------------------------------------------------------------------------
# Individual post-processing steps
# ---------------------------------------------------------------------------

def _is_no_response(text: str, energy: float, config: PostProcessingConfig) -> bool:
    """Detect if participant didn't respond."""
    # Empty or very short
    clean = text.strip()
    if not clean:
        return True

    # Very low energy
    if energy > 0 and energy < config.no_response_energy_threshold:
        return True

    # Only noise markers
    noise_patterns = [r"^\.*$", r"^\.{3,}$", r"^…$"]
    for pat in noise_patterns:
        if re.match(pat, clean):
            return True

    # Less than minimum words
    words = clean.split()
    if len(words) <= config.no_response_min_words and len(clean) < 3:
        return True

    return False


def _remove_hallucinations(
    text: str, stimulus: str, config: PostProcessingConfig,
) -> Tuple[str, bool]:
    """Detect and mitigate Whisper hallucinations.

    Common hallucination patterns:
    - Repeated phrases (same 3+ word sequence appearing twice)
    - Text that's an exact copy of the stimulus (we're transcribing
      the stimulus playback instead of the response)
    """
    was_hallucination = False

    # Check for repeated n-grams
    words = text.split()
    n = config.hallucination_repeat_threshold
    if len(words) >= n * 2:
        ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        repeated = {ng: c for ng, c in counts.items() if c >= 2}
        if repeated:
            # Remove one instance of each repeated n-gram
            for ng in repeated:
                text = text.replace(ng + " " + ng, ng, 1)
            was_hallucination = True
            logger.debug("Removed repeated n-grams: %s", repeated)

    # Check for exact stimulus match
    # (If response == stimulus AND this is unusual, flag it but keep)
    if _normalise_for_comparison(text) == _normalise_for_comparison(stimulus):
        was_hallucination = True
        logger.debug("Transcription matches stimulus exactly — might be stimulus audio")

    return text.strip(), was_hallucination


def _fix_accents(text: str) -> str:
    """Fix common accent/diacritical errors from Whisper.

    Only corrects words that should have accents in standard Spanish.
    Does NOT correct learner errors.
    """
    words = text.split()
    corrected = []

    for word in words:
        lower = word.lower().strip(".,;:!?¿¡\"'()[]")
        if lower in NO_CORRECT_WORDS:
            corrected.append(word)
        elif lower in ACCENT_CORRECTIONS:
            # Preserve original casing pattern
            replacement = ACCENT_CORRECTIONS[lower]
            if word[0].isupper() and not lower[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            # Preserve trailing punctuation
            trailing = ""
            for c in reversed(word):
                if c in ".,;:!?¿¡\"'()[]":
                    trailing = c + trailing
                else:
                    break
            corrected.append(replacement + trailing)
        else:
            corrected.append(word)

    return " ".join(corrected)


def _format_disfluencies(text: str) -> str:
    """Format ASR output disfluencies to match reference conventions.

    Reference conventions (from scoring file):
    - False starts: co-  (word fragment + dash)
    - Pauses: ... or [pause]
    - Unintelligible: xxx or [gibberish]
    - No response: [no response]
    - Self-corrections: word1-word2
    - Uncertain: word(?)
    """
    # Convert multiple silence indicators to [pause]
    text = re.sub(r'\.{4,}', '...', text)

    # Convert "umm", "uhh", "ehh" etc to pause markers
    text = re.sub(r'\b[uU][hm]{1,3}\b', '...', text)
    text = re.sub(r'\b[eE][hm]{1,3}\b', '...', text)
    text = re.sub(r'\b[aA][hm]{1,3}\b', '...', text)

    # Convert "[inaudible]" or similar to xxx
    text = re.sub(r'\[inaudible\]', 'xxx', text, flags=re.IGNORECASE)
    text = re.sub(r'\[unintelligible\]', 'xxx', text, flags=re.IGNORECASE)
    text = re.sub(r'\[indistinct\]', 'xxx', text, flags=re.IGNORECASE)

    # Consolidate multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Consolidate multiple pause markers
    text = re.sub(r'(\.\.\.\s*){2,}', '... ', text)

    return text.strip()


def _cleanup(text: str) -> str:
    """General text cleanup."""
    # Remove leading/trailing whitespace
    text = text.strip()

    # Consolidate spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove weird unicode characters that Whisper sometimes generates
    # but keep standard Spanish characters (ñ, accented vowels, ¿, ¡, etc.)
    allowed_pattern = re.compile(
        r'[^\w\s.,;:!?¿¡\'\"()\[\]\-…ñÑáéíóúÁÉÍÓÚüÜ]',
        re.UNICODE,
    )
    text = allowed_pattern.sub('', text)

    return text.strip()


def _compute_similarity(text1: str, text2: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    words1 = set(_normalise_for_comparison(text1).split())
    words2 = set(_normalise_for_comparison(text2).split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def _normalise_for_comparison(text: str) -> str:
    """Normalise text for comparison (lowercase, remove punctuation)."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def postprocess_all(
    transcriptions: List[Dict[str, Any]],
    stimuli: List[str],
    config: PostProcessingConfig,
) -> List[Dict[str, Any]]:
    """Post-process all transcriptions for one participant.

    Parameters
    ----------
    transcriptions : list of dict
        Each dict has at least: ``raw_text``, ``avg_log_prob``, ``no_speech_prob``,
        ``segment_energy``.
    stimuli : list of str
        The 30 target sentences.
    config : PostProcessingConfig
        Configuration.

    Returns
    -------
    list of dict
        Each dict updated with ``text``, ``flagged``, ``flag_reason``.
    """
    results = []
    for i, trans in enumerate(transcriptions):
        stim = stimuli[i] if i < len(stimuli) else ""
        text, flagged, flag_reason = postprocess_transcription(
            raw_text=trans.get("raw_text", ""),
            stimulus=stim,
            config=config,
            avg_log_prob=trans.get("avg_log_prob", 0.0),
            no_speech_prob=trans.get("no_speech_prob", 0.0),
            segment_energy=trans.get("segment_energy", 0.0),
        )
        result = dict(trans)
        result["text"] = text
        result["flagged"] = flagged
        result["flag_reason"] = flag_reason
        results.append(result)

    return results
