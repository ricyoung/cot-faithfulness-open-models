#!/usr/bin/env python3
"""Deduplicate JSONL result files for the CoT faithfulness project.

For every JSONL file under results/base/ and results/hinted/ for the 12 in-scope models:
1. Read all lines, parse as JSON
2. Group by question_id
3. If duplicates exist, keep the LAST entry per question_id (latest run wins)
4. Write back to the same file
5. Report how many duplicates were removed per file

Also verify after dedup:
- Each base file has exactly 498 unique question_ids (or report gaps)
- Each hinted file has >= 450 unique question_ids (or report gaps)
- Report extraction failure rate (extracted_answer is null) per file
"""

import json
import os
from pathlib import Path
from collections import OrderedDict

BASE_DIR = Path("/Volumes/Data_Remote/_git_remote_long/01-cot-faithfulness-remote")
RESULTS_DIR = BASE_DIR / "results"

IN_SCOPE_MODELS = [
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

HINT_TYPES = ["sycophancy", "consistency", "visual_pattern", "metadata", "grader", "unethical"]

EXPECTED_BASE_COUNT = 498


def dedup_jsonl_file(filepath: Path) -> dict:
    """Deduplicate a single JSONL file, keeping last entry per question_id.

    Returns a dict with stats about the operation.
    """
    if not filepath.exists():
        return {"status": "missing", "path": str(filepath)}

    file_size = filepath.stat().st_size
    if file_size == 0:
        return {"status": "empty", "path": str(filepath)}

    # Read all lines
    records = []
    parse_errors = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                parse_errors += 1

    if not records:
        return {
            "status": "no_valid_records",
            "path": str(filepath),
            "parse_errors": parse_errors,
        }

    total_before = len(records)

    # Group by question_id, keeping last entry (latest run wins)
    seen = OrderedDict()
    for record in records:
        qid = record.get("question_id")
        if qid is None:
            continue
        seen[qid] = record  # overwrite = keep last

    deduped = list(seen.values())
    total_after = len(deduped)
    duplicates_removed = total_before - total_after

    # Count extraction failures
    null_extracted = sum(
        1 for r in deduped
        if r.get("extracted_answer") is None
    )

    # Write back only if duplicates were found
    if duplicates_removed > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            for record in deduped:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "status": "ok",
        "path": str(filepath),
        "total_before": total_before,
        "total_after": total_after,
        "duplicates_removed": duplicates_removed,
        "null_extracted": null_extracted,
        "extraction_failure_rate": (
            f"{null_extracted}/{total_after} ({100*null_extracted/total_after:.1f}%)"
            if total_after > 0 else "N/A"
        ),
        "parse_errors": parse_errors,
    }


def main():
    all_results = []

    # ── Base files ──
    print("=" * 80)
    print("BASE FILES")
    print("=" * 80)

    base_dir = RESULTS_DIR / "base"
    for model in sorted(IN_SCOPE_MODELS):
        filepath = base_dir / f"{model}.jsonl"
        result = dedup_jsonl_file(filepath)
        result["model"] = model
        result["category"] = "base"
        all_results.append(result)

        status = result["status"]
        if status == "missing":
            print(f"  MISSING: {filepath}")
        elif status == "empty":
            print(f"  EMPTY:   {filepath}")
        elif status == "ok":
            dup_str = (
                f"  DEDUPED {result['duplicates_removed']:>3} duplicates"
                if result["duplicates_removed"] > 0
                else "  no duplicates"
            )
            count = result["total_after"]
            gap_str = ""
            if count != EXPECTED_BASE_COUNT:
                gap_str = f"  *** GAP: {count}/{EXPECTED_BASE_COUNT} ***"
            print(
                f"  {model:<30s} {dup_str:>30s}  |  "
                f"{count:>3} unique q_ids  |  "
                f"null_extracted: {result['extraction_failure_rate']}"
                f"{gap_str}"
            )
        else:
            print(f"  {model}: {status}")

    # ── Hinted files ──
    print()
    print("=" * 80)
    print("HINTED FILES")
    print("=" * 80)

    hinted_dir = RESULTS_DIR / "hinted"
    for model in sorted(IN_SCOPE_MODELS):
        model_dir = hinted_dir / model
        if not model_dir.exists():
            print(f"\n  MISSING DIR: {model_dir}")
            continue

        print(f"\n  {model}:")
        for hint in sorted(HINT_TYPES):
            filepath = model_dir / f"{hint}.jsonl"
            result = dedup_jsonl_file(filepath)
            result["model"] = model
            result["hint_type"] = hint
            result["category"] = "hinted"
            all_results.append(result)

            status = result["status"]
            if status == "missing":
                print(f"    {hint:<20s}  MISSING")
            elif status == "empty":
                print(f"    {hint:<20s}  EMPTY")
            elif status == "ok":
                dup_str = (
                    f"deduped {result['duplicates_removed']:>3}"
                    if result["duplicates_removed"] > 0
                    else "no dups"
                )
                count = result["total_after"]
                gap_flag = ""
                if count < 450:
                    gap_flag = f"  *** LOW: {count} < 450 ***"
                print(
                    f"    {hint:<20s}  {dup_str:>12s}  |  "
                    f"{count:>3} unique  |  "
                    f"null_extracted: {result['extraction_failure_rate']}"
                    f"{gap_flag}"
                )
            else:
                print(f"    {hint:<20s}  {status}")

    # ── Summary ──
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    ok_results = [r for r in all_results if r["status"] == "ok"]
    total_dups = sum(r["duplicates_removed"] for r in ok_results)
    files_with_dups = sum(1 for r in ok_results if r["duplicates_removed"] > 0)
    total_files = len(ok_results)
    missing_files = sum(1 for r in all_results if r["status"] == "missing")
    empty_files = sum(1 for r in all_results if r["status"] == "empty")

    print(f"  Files processed:          {total_files}")
    print(f"  Files with duplicates:    {files_with_dups}")
    print(f"  Total duplicates removed: {total_dups}")
    print(f"  Missing files:            {missing_files}")
    print(f"  Empty files:              {empty_files}")

    # Base coverage
    print()
    print("  Base file coverage (expected 498 per model):")
    base_results = [r for r in ok_results if r["category"] == "base"]
    for r in sorted(base_results, key=lambda x: x["model"]):
        count = r["total_after"]
        flag = " OK" if count == EXPECTED_BASE_COUNT else f" *** {count}/{EXPECTED_BASE_COUNT} ***"
        print(f"    {r['model']:<30s} {count:>4}{flag}")

    # Hinted coverage
    print()
    print("  Hinted file coverage (expected >= 450 per file):")
    hinted_results = [r for r in ok_results if r["category"] == "hinted"]
    low_count_files = [r for r in hinted_results if r["total_after"] < 450]
    if low_count_files:
        for r in sorted(low_count_files, key=lambda x: (x["model"], x.get("hint_type", ""))):
            print(
                f"    {r['model']:<25s} {r.get('hint_type',''):<18s} "
                f"{r['total_after']:>4} *** LOW ***"
            )
    else:
        print("    All hinted files have >= 450 unique question_ids.")

    # Extraction failure summary
    print()
    print("  Extraction failure rates (files with > 5% null extracted_answer):")
    high_failure = [
        r for r in ok_results
        if r["total_after"] > 0 and r["null_extracted"] / r["total_after"] > 0.05
    ]
    if high_failure:
        for r in sorted(high_failure, key=lambda x: x["null_extracted"] / max(x["total_after"], 1), reverse=True):
            label = r["model"]
            if r["category"] == "hinted":
                label += f"/{r.get('hint_type', '')}"
            print(f"    {label:<45s} {r['extraction_failure_rate']}")
    else:
        print("    All files have <= 5% extraction failure rate.")

    print()


if __name__ == "__main__":
    main()
