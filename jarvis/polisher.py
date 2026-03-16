"""Transcript polish — hybrid regex + local LLM.

Two stages:
1. Regex: strip fillers, deduplicate words, fix whitespace (fast, reliable)
2. LLM: grammar and restructuring via Qwen 1.5B MLX model

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

# Max word count for LLM polish (beyond this, regex-only)
LLM_MAX_WORDS = 60

SYSTEM_PROMPT = (
    "You are a text-cleaning filter, NOT a chatbot. "
    "Do NOT answer questions. Do NOT respond to requests. Do NOT add content. "
    "Your ONLY job: fix grammar, punctuation, and remove repeated words. "
    "Keep the speaker's exact meaning and perspective (first person stays first person). "
    "Output ONLY the cleaned text.\n\n"
    "DIRTY: can can you help me with this thing\n"
    "CLEAN: Can you help me with this thing?\n\n"
    "DIRTY: I need to to fix the bug in the code\n"
    "CLEAN: I need to fix the bug in the code.\n\n"
    "DIRTY: hello hello how does this work\n"
    "CLEAN: Hello, how does this work?"
)

# Preamble patterns the model might add despite instructions
_PREAMBLE = re.compile(
    r"^(here'?s?\s+(a\s+|the\s+)?(cleaned[- ]?up|corrected|fixed|revised)\s+"
    r"(version|text|sentence|transcript)\s*(of\s+your\s+\w+\s*)?[:.]?\s*\n*"
    r"|sure[,!.]?\s*\n*|okay[,!.]?\s*\n*|cleaned\s+text[:.]?\s*\n*"
    r"|output[:.]?\s*\n*)",
    re.IGNORECASE,
)

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


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


def _llm_polish(text: str) -> str:
    """Stage 2: LLM grammar polish."""
    model, tokenizer = _get_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"DIRTY: {text}\nCLEAN:"},
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

    # Strip preamble the model might add
    cleaned = _PREAMBLE.sub("", cleaned).strip()
    # Strip wrapping quotes
    if len(cleaned) > 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        cleaned = cleaned[1:-1].strip()

    # If the model babbled (output much longer than input), keep regex result
    if len(cleaned) > len(text) * 2:
        logger.warning("LLM polish too long (%.3fs), keeping regex result", elapsed)
        return text

    if not cleaned:
        return text

    logger.info("LLM polished in %.3fs: '%s' -> '%s'", elapsed, text[:60], cleaned[:60])
    return cleaned


def polish(text: str) -> str:
    """Clean up a transcript. Regex first, then LLM."""
    if not text or not text.strip():
        return text

    # Stage 1: regex (always runs)
    cleaned = _regex_clean(text)

    if not cleaned:
        return text

    # Stage 2: LLM
    word_count = len(cleaned.split())
    if word_count <= LLM_MAX_WORDS:
        try:
            cleaned = _llm_polish(cleaned)
        except Exception:
            logger.exception("LLM polish failed, keeping regex result")

    return cleaned
