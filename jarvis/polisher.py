"""Transcript polish — hybrid regex + local LLM.

Two stages:
1. Regex: strip fillers, deduplicate words, fix whitespace (fast, reliable)
2. LLM: grammar and restructuring via small MLX model (only for short/medium inputs
   where 0.5B models are effective — under ~20 words after regex cleanup)

Singleton model with explicit preload for Metal GPU safety.
"""

import logging
import re
import time

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

# Filler words to strip (en + it)
_FILLERS = re.compile(
    r"\b(uh|uhm|um|umm|ah|eh|er|erm|hmm|hm|like|you know|i mean|basically|actually"
    r"|allora|cioè|ehm|beh|mah|diciamo|praticamente)\b",
    re.IGNORECASE,
)

# Max word count for LLM polish (0.5B models degrade on longer inputs)
LLM_MAX_WORDS = 25

SYSTEM_PROMPT = (
    "Clean this transcript. Fix grammar and punctuation. "
    "Output ONLY the cleaned text.\n\n"
    "Examples:\n"
    "Input: can can you help me with this thing\n"
    "Output: Can you help me with this thing?\n\n"
    "Input: I need to to fix the bug in the code\n"
    "Output: I need to fix the bug in the code."
)

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def _get_model(model_name: str | None = None):
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load

        name = model_name or DEFAULT_MODEL
        logger.info("Loading polish model: %s", name)
        _model, _tokenizer = load(name)
        logger.info("Polish model ready")
    return _model, _tokenizer


def preload(model_name: str | None = None):
    """Pre-load model on main thread for safe Metal GPU init."""
    _get_model(model_name)


def _regex_clean(text: str) -> str:
    """Stage 1: fast regex cleanup — fillers, dedup, whitespace."""
    # Strip fillers
    text = _FILLERS.sub("", text)
    # Deduplicate consecutive repeated words: "the the" → "the"
    text = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Fix orphaned punctuation from filler removal
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text


def _word_overlap(a: str, b: str) -> float:
    """Fraction of words in b that also appear in a (case-insensitive)."""
    words_a = set(a.lower().split())
    words_b = b.lower().split()
    if not words_b:
        return 0.0
    return sum(1 for w in words_b if w in words_a) / len(words_b)


def _llm_polish(text: str) -> str:
    """Stage 2: LLM grammar polish for short/medium inputs.

    Includes a divergence guard: if the LLM output shares less than 50%
    of words with the input, it likely hallucinated — fall back to regex.
    """
    model, tokenizer = _get_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    from mlx_lm import generate

    t0 = time.monotonic()
    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max(80, len(text.split()) * 3),
    )
    elapsed = time.monotonic() - t0

    cleaned = result.strip()
    if not cleaned:
        logger.warning("LLM polish returned empty (%.3fs), keeping regex result", elapsed)
        return text

    # Guard: reject if LLM output diverged too much (hallucination)
    overlap = _word_overlap(text, cleaned)
    if overlap < 0.5:
        logger.warning(
            "LLM polish diverged (overlap %.0f%%), keeping regex: '%s' -> '%s'",
            overlap * 100, text[:60], cleaned[:60],
        )
        return text

    logger.info("LLM polished in %.3fs: '%s' -> '%s'", elapsed, text[:60], cleaned[:60])
    return cleaned


def polish(text: str) -> str:
    """Clean up a transcript. Regex first, then LLM for short inputs."""
    if not text or not text.strip():
        return text

    # Stage 1: regex (always runs)
    cleaned = _regex_clean(text)

    if not cleaned:
        return text

    # Stage 2: LLM (only for short/medium inputs where 0.5B is effective)
    word_count = len(cleaned.split())
    if word_count <= LLM_MAX_WORDS:
        try:
            cleaned = _llm_polish(cleaned)
        except Exception:
            logger.exception("LLM polish failed, keeping regex result")

    return cleaned
