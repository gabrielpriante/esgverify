"""Prompt loading with content hashing, for reproducibility.

Prompts are part of the method and are published with the paper, so they live
as plain files in ``scripts/prompts/`` rather than as string literals in the
code. Two consequences follow:

1. A reviewer can read the exact instrument without reading Python.
2. Every extracted record can carry a prompt identifier — ``name@hash`` — so a
   result set is always traceable to the precise wording that produced it. A
   hash of an inline string would change silently whenever the module was
   edited; a hash of a file changes only when the prompt changes.

Prompts are read once and cached. Editing a prompt file requires restarting the
process, which is deliberate: a long run should not silently switch instruments
halfway through.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

# backend/core/pipeline/prompts.py -> repository root
_REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = _REPO_ROOT / "scripts" / "prompts"

# Prompt text sent to an 8B model must stay well clear of the point where the
# effective context collapses. Measured on llama3.1:8b-instruct-q4_K_M: a
# combined prompt of ~1,590 tokens produced word-salad; ~1,240 was fine.
PROMPT_TOKEN_BUDGET = 800


class PromptError(RuntimeError):
    """Raised when a prompt file is missing, empty, or over budget."""


def _approx_tokens(text: str) -> int:
    """Rough token count. Chars/4 is close enough for a budget guard."""
    return len(text) // 4


@lru_cache(maxsize=None)
def load_prompt(name: str) -> tuple[str, str]:
    """Load a prompt file and return ``(text, prompt_id)``.

    Args:
        name: File name inside ``scripts/prompts/``, e.g. ``"detect_v2.txt"``.

    Returns:
        A tuple of the prompt text and an identifier of the form
        ``"<name>@<first 12 hex of sha256>"``, suitable for writing into every
        output record.

    Raises:
        PromptError: If the file is missing, empty, or exceeds
            :data:`PROMPT_TOKEN_BUDGET`.
    """
    path = PROMPT_DIR / name
    if not path.is_file():
        raise PromptError(f"prompt file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise PromptError(f"prompt file is empty: {path}")

    tokens = _approx_tokens(text)
    if tokens > PROMPT_TOKEN_BUDGET:
        raise PromptError(
            f"prompt {name} is ~{tokens} tokens, over the "
            f"{PROMPT_TOKEN_BUDGET} budget. Long prompts collapse the "
            f"effective context window on 8B models and produce word-salad."
        )

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return text, f"{name}@{digest}"


def prompt_id(name: str) -> str:
    """Return just the ``name@hash`` identifier for a prompt file."""
    return load_prompt(name)[1]
