#!/usr/bin/env python3
"""Compare two extraction runs record by record. The real equivalence gate.

Why this exists: the mocked unit tests asserted that concurrency preserves
ordering and merge logic, and they passed — while the live pipeline was
emitting word-salad because the model's effective context had been truncated.
A mock has no context window, so it cannot reproduce that class of failure. Any
change made for speed must therefore be proven against real model output, not
against the suite.

Usage:

    python scripts/compare_runs.py data/samples/before.json data/samples/after.json

Exits 0 when the two runs are equivalent, 1 when they diverge or when either
run is degenerate. Prints a timing comparison either way.

A change that makes the pipeline faster and the output different has not been
validated — it has been traded. This tells you which one happened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Fields compared for equivalence. Deliberately excludes commitment_id (a fresh
# uuid every run) and the provenance fields (expected to differ when the point
# of the comparison is a prompt or model change).
COMPARED_SCALARS = (
    "text",
    "is_commitment",
    "rejection_reason",
    "page_reference",
    "depends_on_outside_factors",
    "restated",
    "is_evidence",
    "verifiability",
)

COMPARED_FIELDS = (
    "target",
    "quantity",
    "deadline",
    "baseline",
    "business_unit",
    "emissions_scope",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", help="Baseline run JSON")
    p.add_argument("after", help="Run to validate against the baseline")
    p.add_argument(
        "--ignore-fields", action="store_true",
        help="Compare only the commitment decision, not the structural fields",
    )
    p.add_argument(
        "--max-report", type=int, default=20,
        help="Stop printing individual differences after this many",
    )
    return p.parse_args()


def load(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "records" not in data:
        raise SystemExit(f"{path} has no 'records' key — not an extraction run")
    return data


def key(record: dict[str, Any]) -> tuple[str, str]:
    """Records are keyed by position and text, not by index.

    Index alignment would silently pair up unrelated records if one run
    produced a different number of them.
    """
    return (str(record.get("page_reference")), " ".join(str(record.get("text", "")).split()))


def field_repr(value: Any) -> str:
    """Render a StructuredField as a comparable, printable string."""
    if not isinstance(value, dict):
        return repr(value)
    status = value.get("status")
    if status == "stated":
        return f"stated:{value.get('value')!r}"
    return str(status)


def degenerate(data: dict[str, Any], label: str) -> bool:
    """Flag a run that parsed nothing — the failure mode that slipped past the
    mocked tests. All-unsure means every model response failed to parse."""
    records = data["records"]
    if not records:
        print(f"  {label}: DEGENERATE — zero records")
        return True
    unsure = sum(1 for r in records if r.get("is_commitment") == "unsure")
    if unsure == len(records):
        print(f"  {label}: DEGENERATE — all {len(records)} records are 'unsure' "
              f"(no model response parsed)")
        return True
    return False


def main() -> int:
    args = parse_args()
    before = load(args.before)
    after = load(args.after)

    print("=" * 78)
    print(f"BEFORE  {Path(args.before).name}")
    print(f"  model={before.get('model')}  records={len(before['records'])}  "
          f"elapsed={before.get('elapsed_seconds')}s  "
          f"chunks={before.get('chunks_processed')}")
    print(f"AFTER   {Path(args.after).name}")
    print(f"  model={after.get('model')}  records={len(after['records'])}  "
          f"elapsed={after.get('elapsed_seconds')}s  "
          f"chunks={after.get('chunks_processed')}")
    print("=" * 78)

    # --- degeneracy check, before anything else ----------------------------
    bad = degenerate(before, "BEFORE") | degenerate(after, "AFTER")
    if bad:
        print("\nFAIL: a degenerate run cannot be compared.")
        return 1

    # --- timing ------------------------------------------------------------
    b_time = before.get("elapsed_seconds")
    a_time = after.get("elapsed_seconds")
    if isinstance(b_time, (int, float)) and isinstance(a_time, (int, float)) and b_time:
        delta = (b_time - a_time) / b_time * 100
        verdict = "faster" if a_time < b_time else "SLOWER"
        print(f"\nTiming: {b_time:.1f}s -> {a_time:.1f}s  ({abs(delta):.0f}% {verdict})")

    # --- alignment ---------------------------------------------------------
    b_map = {key(r): r for r in before["records"]}
    a_map = {key(r): r for r in after["records"]}

    only_before = sorted(set(b_map) - set(a_map))
    only_after = sorted(set(a_map) - set(b_map))
    common = [k for k in (key(r) for r in before["records"]) if k in a_map]

    print(f"\nRecords: {len(common)} aligned, "
          f"{len(only_before)} only in before, {len(only_after)} only in after")

    differences = 0

    for k in only_before:
        if differences < args.max_report:
            print(f"  MISSING in after : [{k[0]}] {k[1][:80]}")
        differences += 1

    for k in only_after:
        if differences < args.max_report:
            print(f"  EXTRA   in after : [{k[0]}] {k[1][:80]}")
        differences += 1

    # --- field-level comparison -------------------------------------------
    for k in common:
        b, a = b_map[k], a_map[k]
        diffs: list[str] = []

        for name in COMPARED_SCALARS:
            if name == "text":
                continue  # part of the key
            if b.get(name) != a.get(name):
                diffs.append(f"{name}: {b.get(name)!r} -> {a.get(name)!r}")

        if not args.ignore_fields:
            for name in COMPARED_FIELDS:
                bf, af = field_repr(b.get(name)), field_repr(a.get(name))
                if bf != af:
                    diffs.append(f"{name}: {bf} -> {af}")

        if diffs:
            if differences < args.max_report:
                print(f"\n  DIFFERS [{k[0]}] {k[1][:80]}")
                for d in diffs:
                    print(f"      {d}")
            differences += len(diffs)

    # --- verdict -----------------------------------------------------------
    print()
    if differences:
        print(f"FAIL: {differences} difference(s). The change altered the output.")
        print("A faster run that answers differently has not been validated.")
        return 1

    b_pos = sum(1 for r in before["records"] if r.get("is_commitment") == "yes")
    b_neg = sum(1 for r in before["records"] if r.get("is_commitment") == "no")
    print(f"PASS: {len(common)} records identical "
          f"({b_pos} commitments, {b_neg} rejections).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
