#!/usr/bin/env python3
"""Bisect why the commitment extractor gets word-salad from a working model.

Context: `ollama run <model> "Name three colors as a JSON array."` returns clean
JSON, but the extractor's /api/chat call on the same model returns degenerate
text. Something between those two calls is responsible. This script varies one
factor at a time and reports which variants produce parseable JSON.

Each variant uses a small num_predict so the whole sweep finishes in a few
minutes even at ~13 tok/s.

Usage (from the repository root):

    python scripts/diagnose_ollama_prompt.py --model llama3.1:8b-instruct-q4_K_M

    # include the real chunk 39 from the Microsoft PDF
    python scripts/diagnose_ollama_prompt.py --model llama3.1:8b-instruct-q4_K_M \\
        --pdf "C:/path/to/The-2023-Impact-Summary.pdf"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from backend.core.config import settings  # noqa: E402
from backend.core.pipeline.commitment_extractor import (  # noqa: E402
    _SYSTEM_PROMPT,
)

SHORT_SYSTEM = """\
You extract environmental commitments from corporate reports.
A commitment is a stated intention to take a FUTURE environmental action or
reach a FUTURE environmental outcome. Past actions are NOT commitments.

Return ONLY JSON:
{"commitments": [{"text": "<sentence>", "is_commitment": "yes|no"}]}
"""

TINY_TEXT = (
    "We will reduce absolute Scope 1 and Scope 2 greenhouse gas emissions "
    "by 50% by 2030, against a 2019 baseline. In 2023, we reduced water "
    "consumption by 12% compared to the prior year."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--pdf", default=None, help="Optional: pull real chunk 39 from this PDF")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--num-predict", type=int, default=400)
    return p.parse_args()


def load_chunk_39(pdf: str) -> str | None:
    try:
        import logging

        import structlog

        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL)
        )
        import fitz

        from backend.core.pipeline.chunker import chunk_text

        parts = []
        with fitz.open(pdf) as doc:
            for page in doc:
                parts.append(page.get_text())
        chunks = chunk_text("\n\n".join(parts))
        for c in chunks:
            if c.index == 39:
                return c.text
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not load chunk 39: {exc})")
    return None


async def call(
    messages: list[dict],
    *,
    model: str,
    timeout: int,
    num_predict: int,
    temperature: float | None,
    fmt_json: bool,
    num_ctx: int | None,
) -> tuple[str, float]:
    options: dict = {"num_predict": num_predict}
    if temperature is not None:
        options["temperature"] = temperature
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    payload: dict = {"model": model, "messages": messages, "stream": False, "options": options}
    if fmt_json:
        payload["format"] = "json"

    started = time.monotonic()
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=timeout) as client:
        r = await client.post("/api/chat", json=payload)
        r.raise_for_status()
    return r.json()["message"]["content"], time.monotonic() - started


def verdict(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")
        ).strip()
    try:
        json.loads(cleaned)
        return "VALID JSON"
    except json.JSONDecodeError:
        return "NOT JSON  "


async def main() -> int:
    args = parse_args()
    chunk39 = load_chunk_39(args.pdf) if args.pdf else None

    sysmsg = lambda s: {"role": "system", "content": s}  # noqa: E731
    usrmsg = lambda s: {"role": "user", "content": s}  # noqa: E731

    variants: list[tuple[str, list[dict], dict]] = [
        # name, messages, kwargs
        ("A control: no system, tiny ask",
         [usrmsg("Name three colors as a JSON array.")],
         dict(temperature=None, fmt_json=False, num_ctx=None)),

        ("B short system + tiny text",
         [sysmsg(SHORT_SYSTEM), usrmsg(TINY_TEXT)],
         dict(temperature=0, fmt_json=True, num_ctx=8192)),

        ("C FULL system + tiny text",
         [sysmsg(_SYSTEM_PROMPT), usrmsg(TINY_TEXT)],
         dict(temperature=0, fmt_json=True, num_ctx=8192)),

        ("D FULL system, temp default",
         [sysmsg(_SYSTEM_PROMPT), usrmsg(TINY_TEXT)],
         dict(temperature=None, fmt_json=True, num_ctx=8192)),

        ("E FULL system, no num_ctx",
         [sysmsg(_SYSTEM_PROMPT), usrmsg(TINY_TEXT)],
         dict(temperature=0, fmt_json=True, num_ctx=None)),

        ("F FULL system inlined as user",
         [usrmsg(_SYSTEM_PROMPT + "\n\n" + TINY_TEXT)],
         dict(temperature=0, fmt_json=True, num_ctx=8192)),
    ]

    if chunk39:
        variants.append(
            ("G FULL system + real chunk 39",
             [sysmsg(_SYSTEM_PROMPT), usrmsg(chunk39)],
             dict(temperature=0, fmt_json=True, num_ctx=8192)))
        variants.append(
            ("H short system + real chunk 39",
             [sysmsg(SHORT_SYSTEM), usrmsg(chunk39)],
             dict(temperature=0, fmt_json=True, num_ctx=8192)))

    print(f"Model: {args.model}   num_predict={args.num_predict}")
    print(f"Chunk 39 loaded: {'yes' if chunk39 else 'no'}")
    print("=" * 72)

    for name, messages, kw in variants:
        try:
            raw, secs = await call(
                messages,
                model=args.model,
                timeout=args.timeout,
                num_predict=args.num_predict,
                **kw,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{name:34s} ERROR      {type(exc).__name__}: {exc}")
            continue

        preview = " ".join(raw.split())[:90]
        print(f"{name:34s} {verdict(raw)} {secs:6.1f}s  {preview}")

    print("=" * 72)
    print("Read: the first variant that flips to NOT JSON is the culprit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
