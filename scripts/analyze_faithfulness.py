#!/usr/bin/env python3
"""Analyze faithfulness from V3 classified results.

Reads results/classified/{model}/{hint_type}.jsonl and produces four CSV files
in results/analysis/:
  1. faithfulness_rate.csv       — per model x hint_type matrix
  2. faithfulness_summary.csv    — per model aggregates
  3. faithfulness_by_hint.csv    — per hint_type aggregates
  4. classification_method_breakdown.csv — regex vs judge counts
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIED_DIR = PROJECT_ROOT / "results" / "classified"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"

MODELS = [
    "deepseek-r1",
    "deepseek-v3.2-speciale",
    "ernie-4.5-21b-thinking",
    "gpt-oss-120b",
    "minimax-m2.5",
    "nemotron-nano-9b",
    "olmo-3-7b-think",
    "olmo-3.1-32b-think",
    "qwen3.5-27b",
    "qwq-32b",
    "seed-1.6-flash",
    "step-3.5-flash",
]

HINT_TYPES = [
    "sycophancy",
    "consistency",
    "visual_pattern",
    "metadata",
    "grader",
    "unethical",
]


def load_records(model: str, hint_type: str) -> list[dict[str, Any]]:
    """Load all JSONL records for a given model and hint type."""
    path = CLASSIFIED_DIR / model / f"{hint_type}.jsonl"
    if not path.exists():
        print(f"  WARNING: Missing file {path}", file=sys.stderr)
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_faithfulness_rate(influenced: list[dict[str, Any]]) -> float:
    """Return faithful / influenced ratio, or 0.0 if no influenced records."""
    if not influenced:
        return 0.0
    faithful_count = sum(1 for r in influenced if r.get("is_faithful") is True)
    return faithful_count / len(influenced)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Accumulate all data keyed by (model, hint_type)
    data: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model in MODELS:
        for ht in HINT_TYPES:
            data[(model, ht)] = load_records(model, ht)

    # ── 1. faithfulness_rate.csv ──────────────────────────────────────────
    rate_path = ANALYSIS_DIR / "faithfulness_rate.csv"
    with open(rate_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + HINT_TYPES + ["avg"])
        for model in MODELS:
            rates: list[float] = []
            for ht in HINT_TYPES:
                influenced = [r for r in data[(model, ht)] if r.get("is_influenced")]
                rate = compute_faithfulness_rate(influenced)
                rates.append(rate)
            avg = sum(rates) / len(rates) if rates else 0.0
            writer.writerow(
                [model] + [f"{r:.4f}" for r in rates] + [f"{avg:.4f}"]
            )
    print(f"Wrote {rate_path}")

    # ── 2. faithfulness_summary.csv ───────────────────────────────────────
    summary_path = ANALYSIS_DIR / "faithfulness_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "total_influenced", "total_faithful",
            "faithfulness_rate", "regex_classified", "judge_classified",
        ])
        for model in MODELS:
            total_influenced = 0
            total_faithful = 0
            regex_count = 0
            judge_count = 0
            for ht in HINT_TYPES:
                for r in data[(model, ht)]:
                    if r.get("is_influenced"):
                        total_influenced += 1
                        if r.get("is_faithful") is True:
                            total_faithful += 1
                        method = r.get("classification_method", "")
                        if method == "regex":
                            regex_count += 1
                        elif method in ("judge", "judge_majority"):
                            judge_count += 1
            rate = total_faithful / total_influenced if total_influenced else 0.0
            writer.writerow([
                model, total_influenced, total_faithful,
                f"{rate:.4f}", regex_count, judge_count,
            ])
    print(f"Wrote {summary_path}")

    # ── 3. faithfulness_by_hint.csv ───────────────────────────────────────
    by_hint_path = ANALYSIS_DIR / "faithfulness_by_hint.csv"
    with open(by_hint_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "hint_type", "total_influenced", "total_faithful", "faithfulness_rate",
        ])
        for ht in HINT_TYPES:
            total_influenced = 0
            total_faithful = 0
            for model in MODELS:
                for r in data[(model, ht)]:
                    if r.get("is_influenced"):
                        total_influenced += 1
                        if r.get("is_faithful") is True:
                            total_faithful += 1
            rate = total_faithful / total_influenced if total_influenced else 0.0
            writer.writerow([ht, total_influenced, total_faithful, f"{rate:.4f}"])
    print(f"Wrote {by_hint_path}")

    # ── 4. classification_method_breakdown.csv ────────────────────────────
    breakdown_path = ANALYSIS_DIR / "classification_method_breakdown.csv"
    with open(breakdown_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "hint_type",
            "regex_faithful", "regex_unfaithful",
            "judge_faithful", "judge_unfaithful",
            "total",
        ])
        for model in MODELS:
            for ht in HINT_TYPES:
                regex_faithful = 0
                regex_unfaithful = 0
                judge_faithful = 0
                judge_unfaithful = 0
                influenced = [r for r in data[(model, ht)] if r.get("is_influenced")]
                for r in influenced:
                    method = r.get("classification_method", "")
                    faithful = r.get("is_faithful") is True
                    if method == "regex":
                        if faithful:
                            regex_faithful += 1
                        else:
                            regex_unfaithful += 1
                    elif method in ("judge", "judge_majority"):
                        if faithful:
                            judge_faithful += 1
                        else:
                            judge_unfaithful += 1
                total = regex_faithful + regex_unfaithful + judge_faithful + judge_unfaithful
                writer.writerow([
                    model, ht,
                    regex_faithful, regex_unfaithful,
                    judge_faithful, judge_unfaithful,
                    total,
                ])
    print(f"Wrote {breakdown_path}")

    # ── Print summary tables to stdout ────────────────────────────────────
    print("\n" + "=" * 80)
    print("FAITHFULNESS RATE BY MODEL x HINT TYPE")
    print("=" * 80)

    # Header
    col_w = 12
    model_w = 28
    header = f"{'Model':<{model_w}}"
    for ht in HINT_TYPES:
        header += f"{ht:>{col_w}}"
    header += f"{'avg':>{col_w}}"
    print(header)
    print("-" * len(header))

    for model in MODELS:
        row = f"{model:<{model_w}}"
        rates: list[float] = []
        for ht in HINT_TYPES:
            influenced = [r for r in data[(model, ht)] if r.get("is_influenced")]
            rate = compute_faithfulness_rate(influenced)
            rates.append(rate)
            row += f"{rate:>{col_w}.1%}"
        avg = sum(rates) / len(rates) if rates else 0.0
        row += f"{avg:>{col_w}.1%}"
        print(row)

    print("\n" + "=" * 80)
    print("FAITHFULNESS SUMMARY BY MODEL")
    print("=" * 80)
    print(f"{'Model':<{model_w}}{'Influenced':>12}{'Faithful':>12}{'Rate':>10}{'Regex':>8}{'Judge':>8}")
    print("-" * (model_w + 50))

    for model in MODELS:
        total_influenced = 0
        total_faithful = 0
        regex_count = 0
        judge_count = 0
        for ht in HINT_TYPES:
            for r in data[(model, ht)]:
                if r.get("is_influenced"):
                    total_influenced += 1
                    if r.get("is_faithful") is True:
                        total_faithful += 1
                    method = r.get("classification_method", "")
                    if method == "regex":
                        regex_count += 1
                    elif method in ("judge", "judge_majority"):
                        judge_count += 1
        rate = total_faithful / total_influenced if total_influenced else 0.0
        print(f"{model:<{model_w}}{total_influenced:>12}{total_faithful:>12}{rate:>10.1%}{regex_count:>8}{judge_count:>8}")

    print("\n" + "=" * 80)
    print("FAITHFULNESS BY HINT TYPE")
    print("=" * 80)
    print(f"{'Hint Type':<20}{'Influenced':>12}{'Faithful':>12}{'Rate':>10}")
    print("-" * 54)
    for ht in HINT_TYPES:
        total_influenced = 0
        total_faithful = 0
        for model in MODELS:
            for r in data[(model, ht)]:
                if r.get("is_influenced"):
                    total_influenced += 1
                    if r.get("is_faithful") is True:
                        total_faithful += 1
        rate = total_faithful / total_influenced if total_influenced else 0.0
        print(f"{ht:<20}{total_influenced:>12}{total_faithful:>12}{rate:>10.1%}")


if __name__ == "__main__":
    main()
