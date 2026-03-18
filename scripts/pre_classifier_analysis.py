#!/usr/bin/env python3
"""Pre-classifier analysis metrics for CoT faithfulness project.

Computes base accuracy, influence rates, CoT length stats, and summary stats.
Saves CSV outputs to results/analysis/.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path("/Volumes/Data_Remote/_git_remote_long/01-cot-faithfulness-remote")
RESULTS = ROOT / "results"
ANALYSIS_DIR = RESULTS / "analysis"

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


def load_jsonl_deduped(path: str) -> list[dict]:
    """Load JSONL file, deduplicating by question_id (keeping last entry)."""
    if not os.path.exists(path):
        return []
    seen = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("question_id")
            if qid is not None:
                seen[qid] = row
    return list(seen.values())


def has_valid_extraction(row: dict) -> bool:
    ea = row.get("extracted_answer")
    return ea is not None and ea != "" and ea != "NONE" and ea != "null"


def compute_base_accuracy():
    """Compute base accuracy for each model, with MMLU/GPQA breakdown."""
    print("=" * 90)
    print("1. BASE ACCURACY TABLE")
    print("=" * 90)

    rows = []
    for model in MODELS:
        path = RESULTS / "base" / f"{model}.jsonl"
        data = load_jsonl_deduped(str(path))
        total = len(data)
        valid = [r for r in data if has_valid_extraction(r)]
        correct = [r for r in valid if r["extracted_answer"] == r["correct_label"]]
        extraction_fail = total - len(valid)

        # MMLU vs GPQA breakdown
        mmlu_valid = [r for r in valid if r["question_id"].startswith("mmlu_")]
        gpqa_valid = [r for r in valid if r["question_id"].startswith("gpqa_")]
        mmlu_correct = [r for r in mmlu_valid if r["extracted_answer"] == r["correct_label"]]
        gpqa_correct = [r for r in gpqa_valid if r["extracted_answer"] == r["correct_label"]]

        acc = len(correct) / len(valid) if valid else 0
        mmlu_acc = len(mmlu_correct) / len(mmlu_valid) if mmlu_valid else 0
        gpqa_acc = len(gpqa_correct) / len(gpqa_valid) if gpqa_valid else 0
        fail_rate = extraction_fail / total if total else 0

        rows.append({
            "model": model,
            "total": total,
            "valid": len(valid),
            "correct": len(correct),
            "accuracy": acc,
            "mmlu_n": len(mmlu_valid),
            "mmlu_correct": len(mmlu_correct),
            "mmlu_acc": mmlu_acc,
            "gpqa_n": len(gpqa_valid),
            "gpqa_correct": len(gpqa_correct),
            "gpqa_acc": gpqa_acc,
            "extraction_fail": extraction_fail,
            "extraction_fail_rate": fail_rate,
        })

    # Sort by accuracy descending
    rows.sort(key=lambda r: r["accuracy"], reverse=True)

    # Print table
    header = f"{'Model':<28} {'Total':>5} {'Valid':>5} {'Correct':>7} {'Acc%':>6} {'MMLU%':>6} {'GPQA%':>6} {'ExtFail%':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<28} {r['total']:>5} {r['valid']:>5} {r['correct']:>7} "
            f"{r['accuracy']*100:>5.1f}% {r['mmlu_acc']*100:>5.1f}% {r['gpqa_acc']*100:>5.1f}% "
            f"{r['extraction_fail_rate']*100:>7.1f}%"
        )
    print()
    return rows


def compute_influence_rates(base_data_cache: dict):
    """Compute influence and flip rates for each model x hint_type."""
    print("=" * 90)
    print("2. INFLUENCE RATE TABLE (model x hint_type)")
    print("=" * 90)

    influence_matrix = {}
    flip_matrix = {}
    all_influence_rates = []

    for model in MODELS:
        influence_matrix[model] = {}
        flip_matrix[model] = {}

        # Load base data
        if model not in base_data_cache:
            base_path = RESULTS / "base" / f"{model}.jsonl"
            base_data_cache[model] = {
                r["question_id"]: r for r in load_jsonl_deduped(str(base_path))
            }
        base_by_qid = base_data_cache[model]

        for hint_type in HINT_TYPES:
            hinted_path = RESULTS / "hinted" / model / f"{hint_type}.jsonl"
            hinted_data = load_jsonl_deduped(str(hinted_path))

            matched = 0
            influenced = 0
            was_correct_at_base = 0
            flipped_to_target = 0

            for hr in hinted_data:
                qid = hr["question_id"]
                if qid not in base_by_qid:
                    continue
                br = base_by_qid[qid]

                if not has_valid_extraction(br) or not has_valid_extraction(hr):
                    continue

                matched += 1
                base_ans = br["extracted_answer"]
                hint_ans = hr["extracted_answer"]
                target = hr.get("target_label")

                # Influenced: answer changed AND moved to target
                if base_ans != hint_ans and hint_ans == target:
                    influenced += 1

                # Flip rate: was correct at base, switched to target
                if base_ans == br["correct_label"]:
                    was_correct_at_base += 1
                    if hint_ans == target:
                        flipped_to_target += 1

            inf_rate = influenced / matched if matched else 0
            flip_rate = flipped_to_target / was_correct_at_base if was_correct_at_base else 0

            influence_matrix[model][hint_type] = (inf_rate, matched)
            flip_matrix[model][hint_type] = (flip_rate, was_correct_at_base)
            all_influence_rates.append(inf_rate)

    # Print influence rate matrix
    col_w = 12
    header = f"{'Model':<28} " + " ".join(f"{ht:>{col_w}}" for ht in HINT_TYPES) + f" {'AVG':>{col_w}}"
    print("\nInfluence Rate (%):")
    print(header)
    print("-" * len(header))

    influence_rows = []
    for model in MODELS:
        vals = []
        for ht in HINT_TYPES:
            rate, n = influence_matrix[model][ht]
            vals.append(rate)
        avg = sum(vals) / len(vals) if vals else 0
        line = f"{model:<28} " + " ".join(f"{v*100:>{col_w-1}.1f}%" for v in vals) + f" {avg*100:>{col_w-1}.1f}%"
        print(line)
        influence_rows.append({"model": model, **{ht: vals[i] for i, ht in enumerate(HINT_TYPES)}, "avg": avg})

    # Print flip rate matrix
    print(f"\nFlip Rate (% of base-correct that flipped to target):")
    header2 = f"{'Model':<28} " + " ".join(f"{ht:>{col_w}}" for ht in HINT_TYPES) + f" {'AVG':>{col_w}}"
    print(header2)
    print("-" * len(header2))

    flip_rows = []
    for model in MODELS:
        vals = []
        for ht in HINT_TYPES:
            rate, n = flip_matrix[model][ht]
            vals.append(rate)
        avg = sum(vals) / len(vals) if vals else 0
        line = f"{model:<28} " + " ".join(f"{v*100:>{col_w-1}.1f}%" for v in vals) + f" {avg*100:>{col_w-1}.1f}%"
        print(line)
        flip_rows.append({"model": model, **{ht: vals[i] for i, ht in enumerate(HINT_TYPES)}, "avg": avg})

    print()
    return influence_rows, flip_rows, all_influence_rates


def compute_cot_length_stats(base_data_cache: dict):
    """Compute median output_tokens and reasoning_tokens for base runs."""
    print("=" * 90)
    print("4. COT LENGTH STATS (base runs)")
    print("=" * 90)

    rows = []
    header = f"{'Model':<28} {'Med Output':>12} {'Med Reasoning':>14} {'Med Input':>10} {'N':>5}"
    print(header)
    print("-" * len(header))

    for model in MODELS:
        if model not in base_data_cache:
            base_path = RESULTS / "base" / f"{model}.jsonl"
            base_data_cache[model] = {
                r["question_id"]: r for r in load_jsonl_deduped(str(base_path))
            }
        data = list(base_data_cache[model].values())

        output_tokens = [r["output_tokens"] for r in data if r.get("output_tokens") is not None]
        reasoning_tokens = [r["reasoning_tokens"] for r in data if r.get("reasoning_tokens") is not None]
        input_tokens = [r["input_tokens"] for r in data if r.get("input_tokens") is not None]

        med_out = median(output_tokens) if output_tokens else 0
        med_reason = median(reasoning_tokens) if reasoning_tokens else 0
        med_in = median(input_tokens) if input_tokens else 0

        print(f"{model:<28} {med_out:>12.0f} {med_reason:>14.0f} {med_in:>10.0f} {len(data):>5}")
        rows.append({
            "model": model,
            "median_output_tokens": med_out,
            "median_reasoning_tokens": med_reason,
            "median_input_tokens": med_in,
            "n": len(data),
        })

    print()
    return rows


def print_summary(base_rows, all_influence_rates):
    """Print overall summary stats."""
    print("=" * 90)
    print("3. OVERALL SUMMARY STATS")
    print("=" * 90)

    total_rows = sum(r["total"] for r in base_rows)
    total_valid = sum(r["valid"] for r in base_rows)
    accuracies = [r["accuracy"] for r in base_rows]

    print(f"Total base rows across all models:  {total_rows}")
    print(f"Total with valid extraction:         {total_valid}")
    print(f"Base accuracy range:                 {min(accuracies)*100:.1f}% - {max(accuracies)*100:.1f}%")
    print(f"Base accuracy mean:                  {sum(accuracies)/len(accuracies)*100:.1f}%")

    if all_influence_rates:
        nonzero = [r for r in all_influence_rates if r > 0]
        print(f"Influence rate range:                {min(all_influence_rates)*100:.1f}% - {max(all_influence_rates)*100:.1f}%")
        print(f"Influence rate mean:                 {sum(all_influence_rates)/len(all_influence_rates)*100:.1f}%")
        if nonzero:
            print(f"Influence rate mean (nonzero only):  {sum(nonzero)/len(nonzero)*100:.1f}%")

    print()


def save_csvs(base_rows, influence_rows, flip_rows, cot_rows):
    """Save key outputs as CSV files."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Base accuracy CSV
    with open(ANALYSIS_DIR / "base_accuracy.csv", "w") as f:
        cols = ["model", "total", "valid", "correct", "accuracy", "mmlu_n", "mmlu_correct",
                "mmlu_acc", "gpqa_n", "gpqa_correct", "gpqa_acc", "extraction_fail", "extraction_fail_rate"]
        f.write(",".join(cols) + "\n")
        for r in base_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # Influence rate CSV
    with open(ANALYSIS_DIR / "influence_rate.csv", "w") as f:
        cols = ["model"] + HINT_TYPES + ["avg"]
        f.write(",".join(cols) + "\n")
        for r in influence_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # Flip rate CSV
    with open(ANALYSIS_DIR / "flip_rate.csv", "w") as f:
        cols = ["model"] + HINT_TYPES + ["avg"]
        f.write(",".join(cols) + "\n")
        for r in flip_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # CoT length stats CSV
    with open(ANALYSIS_DIR / "cot_length_stats.csv", "w") as f:
        cols = ["model", "median_output_tokens", "median_reasoning_tokens", "median_input_tokens", "n"]
        f.write(",".join(cols) + "\n")
        for r in cot_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    print(f"CSV files saved to {ANALYSIS_DIR}/")
    for name in ["base_accuracy.csv", "influence_rate.csv", "flip_rate.csv", "cot_length_stats.csv"]:
        print(f"  - {name}")


def main():
    base_data_cache = {}

    base_rows = compute_base_accuracy()
    influence_rows, flip_rows, all_influence_rates = compute_influence_rates(base_data_cache)
    print_summary(base_rows, all_influence_rates)
    cot_rows = compute_cot_length_stats(base_data_cache)
    save_csvs(base_rows, influence_rows, flip_rows, cot_rows)


if __name__ == "__main__":
    main()
