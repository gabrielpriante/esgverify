#!/usr/bin/env python3
"""Batch commitment extraction across the full corpus, resumable across nights.

Designed for a multi-session overnight run that may crash, be interrupted, or
be stopped deliberately. Restarting is always safe: a document whose output
already exists is skipped, never redone and never overwritten.

RESUME MATCHES ON DOCUMENT, NOT ON DATE. The single-document runner names its
output <document>_<YYYY-MM-DD>.json. If the resume check looked for today's
date, then every document processed on night 1 would look unprocessed on night
2 and the whole corpus would be redone nightly, forever. So the check globs
<document>_*.json and treats any dated match as done.

Each document runs in its own subprocess. In-process would be marginally
faster, but a batch meant to survive unattended needs isolation: a segfault in
a PDF parser, a memory leak, or a hung request kills one document instead of
the night's work. Subprocess startup is a couple of seconds against tens of
minutes of inference.

SINGLE SLOT ONLY. Concurrency is pinned to 1 and OLLAMA_NUM_PARALLEL must stay
unset. Parallel slots divide the context window between them, silently
truncating the prompt and producing degenerate output — verified on this
hardware, see notes/decisions.md 07/26/2026.

Usage:

    # see what would run, without calling the model
    python scripts/run_corpus_batch.py --dry-run

    # prove it works on one document first
    python scripts/run_corpus_batch.py --limit-docs 1

    # the overnight run
    python scripts/run_corpus_batch.py 2>&1 | Tee-Object -FilePath logs\\batch.log
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_commitment_extraction.py"

DEFAULT_CORPUS = Path(
    r"C:\Users\gabpe\OneDrive\Documents\publication\ESGVerify-Bench\data\raw"
)

# Guideline-development documents. These are NOT part of the corpus frame:
# the annotation guideline was built on them, so including them would mean
# reporting results on documents the instrument was fitted to.
EXCLUDED_DIRS = {"microsoft", "jnj"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS),
                   help="Root of the corpus; every company subfolder is walked")
    p.add_argument("--out-dir", default="data/corpus_runs",
                   help="Output directory, kept separate from data/samples so "
                        "development runs and corpus runs never mix")
    p.add_argument("--log", default=None,
                   help="Log file path (default logs/corpus_batch_<date>.log)")
    p.add_argument("--model", default="llama3.1:8b-instruct-q4_K_M")
    p.add_argument("--timeout", type=int, default=900,
                   help="Per-request timeout passed to the runner")
    p.add_argument("--doc-timeout", type=int, default=10800,
                   help="Hard ceiling per document in seconds (default 3h). A "
                        "hung document is killed so the batch can continue.")
    p.add_argument("--limit-docs", type=int, default=None,
                   help="Process at most N documents this session")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would run. Makes no model calls.")
    p.add_argument("--exclude", default=",".join(sorted(EXCLUDED_DIRS)),
                   help="Comma-separated folder names to skip (case-insensitive)")
    return p.parse_args()


class Tee:
    """Write progress to console and log file at once.

    Line-buffered and flushed after every write: if the machine dies at 4am,
    the log must already contain everything up to that moment.
    """

    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = log_path.open("a", encoding="utf-8")

    def __call__(self, message: str = "") -> None:
        print(message, flush=True)
        self.fh.write(message + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def discover(corpus_dir: Path, excluded: set[str]) -> list[Path]:
    """Every PDF in the corpus, excluding development documents.

    Matches .pdf case-insensitively — at least one file in this corpus is
    named .PDF, and a case-sensitive glob would silently drop it.
    """
    pdfs = [
        p for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf"
    ]
    kept = [
        p for p in pdfs
        if not any(part.lower() in excluded for part in p.relative_to(corpus_dir).parts)
    ]
    return sorted(kept)


def existing_output(out_dir: Path, pdf: Path) -> Path | None:
    """Return a previous run's output for this document, if any.

    Globs <stem>_*.json rather than testing today's filename, so a resume on a
    later night recognises work done on an earlier one.
    """
    matches = sorted(out_dir.glob(f"{pdf.stem}_*.json"))
    return matches[0] if matches else None


def count_records(path: Path) -> tuple[int, int]:
    """Return (total records, commitments) from a result file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data.get("records", [])
        positives = sum(1 for r in records if r.get("is_commitment") == "yes")
        return len(records), positives
    except (OSError, json.JSONDecodeError):
        return 0, 0


def run_one(pdf: Path, out_path: Path, args: argparse.Namespace) -> tuple[bool, str]:
    """Run the single-document extractor in a subprocess.

    Returns (succeeded, detail). Never raises: a failure here must not end the
    batch.
    """
    cmd = [
        sys.executable, str(RUNNER),
        "--pdf", str(pdf),
        "--out", str(out_path),
        "--model", args.model,
        "--timeout", str(args.timeout),
        "--concurrency", "1",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
            timeout=args.doc_timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"exceeded --doc-timeout of {args.doc_timeout}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit code {proc.returncode}"
        return False, detail[:200]

    if not out_path.is_file():
        return False, "runner reported success but wrote no output file"

    return True, ""


def main() -> int:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)
    excluded = {e.strip().lower() for e in args.exclude.split(",") if e.strip()}

    log_path = Path(args.log) if args.log else Path(
        f"logs/corpus_batch_{date.today().isoformat()}.log"
    )
    say = Tee(log_path)

    if not corpus_dir.is_dir():
        say(f"ERROR: corpus directory not found: {corpus_dir}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    say("=" * 78)
    say(f"CORPUS BATCH  session started {datetime.now():%Y-%m-%d %H:%M:%S}")
    say(f"  corpus   {corpus_dir}")
    say(f"  output   {out_dir}")
    say(f"  log      {log_path}")
    say(f"  model    {args.model}   concurrency=1 (single slot)")
    say(f"  excluded {sorted(excluded)}")
    say("=" * 78)

    all_pdfs = discover(corpus_dir, excluded)
    todo = [p for p in all_pdfs if existing_output(out_dir, p) is None]
    already = len(all_pdfs) - len(todo)

    say(f"{len(all_pdfs)} corpus documents, {already} already done, {len(todo)} remaining")

    if args.limit_docs is not None:
        todo = todo[: args.limit_docs]
        say(f"--limit-docs {args.limit_docs}: processing {len(todo)} this session")

    if args.dry_run:
        say("\nDRY RUN — no model calls will be made.\n")
        for i, pdf in enumerate(todo, 1):
            say(f"  [{i:3d}] would process  {pdf.relative_to(corpus_dir)}")
        skipped_preview = [p for p in all_pdfs if existing_output(out_dir, p)]
        for pdf in skipped_preview[:10]:
            found = existing_output(out_dir, pdf)
            say(f"        would SKIP     {pdf.relative_to(corpus_dir)}  ({found.name})")
        if len(skipped_preview) > 10:
            say(f"        ... and {len(skipped_preview) - 10} more already done")
        say(f"\nDry run complete. {len(todo)} to process, {already} to skip.")
        say.close()
        return 0

    processed = 0
    failed: list[tuple[str, str]] = []
    total_records = 0
    total_commitments = 0
    session_start = time.monotonic()

    for i, pdf in enumerate(todo, 1):
        rel = pdf.relative_to(corpus_dir)
        out_path = out_dir / f"{pdf.stem}_{date.today().isoformat()}.json"

        # Re-check immediately before running. A long batch can outlive the
        # date it started on, and a parallel session may have claimed this one.
        found = existing_output(out_dir, pdf)
        if found is not None:
            say(f"[{i:3d}/{len(todo)}] SKIP  {rel}  (already {found.name})")
            continue

        say(f"[{i:3d}/{len(todo)}] START {rel}  {datetime.now():%H:%M:%S}")
        started = time.monotonic()
        ok, detail = run_one(pdf, out_path, args)
        elapsed = time.monotonic() - started

        if ok:
            records, commitments = count_records(out_path)
            total_records += records
            total_commitments += commitments
            processed += 1
            say(f"[{i:3d}/{len(todo)}] DONE  {rel}  "
                f"{records} records, {commitments} commitments, {elapsed / 60:.1f} min")
        else:
            failed.append((str(rel), detail))
            say(f"[{i:3d}/{len(todo)}] FAIL  {rel}  after {elapsed / 60:.1f} min — {detail}")

        done_so_far = processed + len(failed)
        if done_so_far and done_so_far < len(todo):
            avg = (time.monotonic() - session_start) / done_so_far
            remaining = (len(todo) - done_so_far) * avg / 3600
            say(f"          pace {avg / 60:.1f} min/doc, "
                f"~{remaining:.1f}h left for this session")

    total_min = (time.monotonic() - session_start) / 60
    say("")
    say("=" * 78)
    say(f"SESSION SUMMARY  finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    say(f"  processed        {processed}")
    say(f"  skipped (done)   {already}")
    say(f"  failed           {len(failed)}")
    say(f"  records          {total_records} ({total_commitments} commitments)")
    say(f"  session time     {total_min / 60:.1f}h")
    remaining_total = len(all_pdfs) - already - processed
    say(f"  corpus remaining {remaining_total} of {len(all_pdfs)}")
    if failed:
        say("\n  FAILED DOCUMENTS — rerun the batch and it will retry these:")
        for name, detail in failed:
            say(f"    {name}\n        {detail}")
    say("=" * 78)
    say.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
