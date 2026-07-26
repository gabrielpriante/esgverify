#!/usr/bin/env python3
"""Run commitment extraction over a single PDF against a local Ollama server.

This is the live pass. It requires Ollama running on this machine and cannot be
executed in a sandbox, which is why it exists as a standalone script rather than
as part of the offline test suite.

Usage (from the repository root):

    python -m scripts.run_commitment_extraction \\
        --pdf "/path/to/The-2023-Impact-Summary.pdf" \\
        --out data/samples/microsoft_2023_commitments.json

Options:
    --limit N     Process only the first N chunks. Start small — a 22 MB report
                  is hundreds of chunks and a full pass takes a long time on an
                  8 GB card.
    --model NAME  Override OLLAMA_MODEL for this run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Allow running as a plain script from the repo root as well as via -m
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.config import settings  # noqa: E402
from backend.core.pipeline.chunker import chunk_text  # noqa: E402
from backend.core.pipeline.commitment_extractor import (  # noqa: E402
    extract_commitments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="Path to the PDF to process")
    parser.add_argument("--out", required=True, help="Path for the JSON output")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only N chunks (recommended for a first run)",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help=(
            "Zero-based chunk index to start from (default 0). The opening "
            "chunks of a report are usually narrative; the commitments live "
            "further in. Combine with --limit to target a specific range."
        ),
    )
    parser.add_argument("--model", default=None, help="Override the Ollama model")
    parser.add_argument(
        "--timeout", type=int, default=600,
        help=(
            "Per-request timeout in seconds (default 600). The config default "
            "of 120 is too short for this prompt: the system prompt is long and "
            "num_predict is 2048, so a single chunk can take several minutes on "
            "a mid-range GPU — and longer still if Ollama falls back to CPU."
        ),
    )
    parser.add_argument(
        "--num-predict", type=int, default=None,
        help=(
            "Override BOTH stages' output cap with one value. Use "
            "--num-predict 2048 to reproduce the old single-cap behaviour for "
            "a baseline comparison."
        ),
    )
    parser.add_argument(
        "--detect-num-predict", type=int, default=None,
        help="Output cap for stage 1 detection (default 800)",
    )
    parser.add_argument(
        "--enrich-num-predict", type=int, default=None,
        help="Output cap for stage 2 enrichment (default 500)",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=None,
        help=(
            "Override the context window (default 8192). Must exceed prompt "
            "plus num-predict, or the model emits word-salad instead of JSON."
        ),
    )
    parser.add_argument(
        "--dump-raw", default=None, metavar="DIR",
        help="Write every raw model response to DIR/chunk_<n>.txt before parsing",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help=(
            "Requests in flight at once (default from config, 4). Ollama "
            "serialises per model unless OLLAMA_NUM_PARALLEL is also raised, "
            "so set both. Each parallel slot needs its own KV cache at "
            "num_ctx, so watch VRAM."
        ),
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"ERROR: no such file: {pdf_path}", file=sys.stderr)
        return 1

    if args.model:
        settings.ollama_model = args.model
    settings.ollama_timeout_seconds = args.timeout

    from backend.core.pipeline import commitment_extractor
    if args.num_predict is not None:
        commitment_extractor.NUM_PREDICT = args.num_predict
    if args.detect_num_predict is not None:
        commitment_extractor.DETECT_NUM_PREDICT = args.detect_num_predict
    if args.enrich_num_predict is not None:
        commitment_extractor.ENRICH_NUM_PREDICT = args.enrich_num_predict
    if args.num_ctx is not None:
        commitment_extractor.NUM_CTX = args.num_ctx
    if args.dump_raw:
        commitment_extractor.RAW_DUMP_DIR = args.dump_raw
    if args.concurrency is not None:
        commitment_extractor.CONCURRENCY = args.concurrency

    print(f"Model:   {settings.ollama_model}")
    print(f"Ollama:  {settings.ollama_base_url}")
    print(f"Timeout: {settings.ollama_timeout_seconds}s per request")
    _override = commitment_extractor.NUM_PREDICT
    print(f"Context: num_ctx={commitment_extractor.NUM_CTX}")
    print(f"Output:  detect={_override or commitment_extractor.DETECT_NUM_PREDICT}, "
          f"enrich={_override or commitment_extractor.ENRICH_NUM_PREDICT} tokens"
          + ("  (single-cap override)" if _override else ""))
    print(f"Parallel:{commitment_extractor.CONCURRENCY} requests in flight")
    print(f"PDF:     {pdf_path.name}")

    # --- 1. PDF -> text ----------------------------------------------------
    import fitz  # pymupdf

    text_parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        for page in doc:
            text_parts.append(page.get_text())
    text = "\n\n".join(text_parts)
    print(f"Parsed {page_count} pages, {len(text):,} characters")

    # --- 2. text -> chunks -------------------------------------------------
    chunks = chunk_text(text)
    total_chunks = len(chunks)
    chunks = chunks[args.start:]
    if args.limit is not None:
        chunks = chunks[: args.limit]
    if chunks:
        span = f"{chunks[0].index}-{chunks[-1].index}"
    else:
        span = "none"
    print(f"Chunked into {total_chunks} chunks; processing {len(chunks)} (index {span})")

    if not chunks:
        print("ERROR: no chunks produced — check PDF text extraction", file=sys.stderr)
        return 1

    # --- 3. chunks -> commitment records -----------------------------------
    started = time.monotonic()
    records = await extract_commitments(chunks)
    elapsed = time.monotonic() - started

    # --- 4. write output ---------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "source_pdf": pdf_path.name,
                "model": settings.ollama_model,
                "pages": page_count,
                "chunks_total": total_chunks,
                "chunks_processed": len(chunks),
                "elapsed_seconds": round(elapsed, 1),
                "records": [r.model_dump(mode="json") for r in records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- 5. summary --------------------------------------------------------
    decisions = Counter(r.is_commitment.value for r in records)
    levels = Counter(
        r.verifiability.value for r in records if r.is_commitment.value == "yes"
    )
    conditional = sum(
        1 for r in records
        if r.is_commitment.value == "yes"
        and r.depends_on_outside_factors.value == "yes"
    )
    restated = sum(
        1 for r in records
        if r.is_commitment.value == "yes" and r.restated.value == "yes"
    )
    evidence = sum(1 for r in records if r.is_evidence.value == "yes")

    print(f"\nDone in {elapsed:.1f}s -> {out_path}")
    print(f"Records: {len(records)}")
    print(f"  yes/no/unsure:        {dict(decisions)}")
    print(f"  verifiability (yes):  {dict(levels)}")
    print(f"  conditional:          {conditional}")
    print(f"  restated:             {restated}")
    print(f"  evidence (not commit):{evidence}")

    if not records:
        print(
            "\nNOTE: zero records. Check that Ollama is running and the model "
            "is pulled, then re-run with --limit 3 and inspect the logs.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
