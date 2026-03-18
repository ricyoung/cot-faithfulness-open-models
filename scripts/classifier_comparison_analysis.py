#!/usr/bin/env python3
"""Classifier comparison analysis for Paper 2.

"Measuring Faithfulness Depends on How You Measure"

Compares two faithfulness classifiers:
  - "Our pipeline": two-stage regex + LLM judge (V3 classifier)
  - "Sonnet judge": single-pass Claude Sonnet judge (sonnet_verdict)

Both classifiers were run on the same influenced cases. Per-case verdicts
are stored in results/classified_sonnet/{model}/{hint_type}.jsonl, where
each record has both `our_verdict` (bool) and `sonnet_verdict` (bool).

Analyses produced:
  1. Per-case confusion matrices by hint type (our x sonnet)
  2. Cohen's kappa per hint type
  3. Regex-only baseline faithfulness rates
  4. Spearman rank correlation (model ranking stability)
  5. Overall and per-hint-type agreement statistics

All tables are printed to stdout and saved as CSVs in results/analysis/paper2/.

Usage:
    python3 scripts/classifier_comparison_analysis.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIED_SONNET_DIR = PROJECT_ROOT / "results" / "classified_sonnet"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
PAPER2_DIR = ANALYSIS_DIR / "paper2"

# ---------------------------------------------------------------------------
# Constants: models and hint types present in classified_sonnet/
# Visual_pattern is excluded from Sonnet comparison (not judged by Sonnet).
# ---------------------------------------------------------------------------

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

# Hint types covered by Sonnet judge (visual_pattern was not sent to Sonnet)
SONNET_HINT_TYPES = [
    "sycophancy",
    "consistency",
    "metadata",
    "grader",
    "unethical",
]

ALL_HINT_TYPES = [
    "sycophancy",
    "consistency",
    "visual_pattern",
    "metadata",
    "grader",
    "unethical",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ConfusionCounts(NamedTuple):
    """2x2 confusion matrix: our classifier (rows) vs Sonnet (cols)."""
    both_faithful: int      # our=T, sonnet=T
    our_only: int           # our=T, sonnet=F
    sonnet_only: int        # our=F, sonnet=T
    both_unfaithful: int    # our=F, sonnet=F
    total: int


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file; return empty list if file doesn't exist."""
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def read_csv_as_dicts(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dicts."""
    if not path.exists():
        print(f"  WARNING: {path} not found", file=sys.stderr)
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write rows to a CSV file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Statistics: Cohen's kappa and Spearman's rho (no scipy required)
# ---------------------------------------------------------------------------

def cohens_kappa(cm: ConfusionCounts) -> float:
    """Compute Cohen's kappa from a 2x2 confusion matrix.

    kappa = (p_o - p_e) / (1 - p_e)

    where:
      p_o = observed agreement = (both_faithful + both_unfaithful) / total
      p_e = expected agreement by chance
            = P(our=T)*P(son=T) + P(our=F)*P(son=F)
    """
    n = cm.total
    if n == 0:
        return float("nan")

    p_o = (cm.both_faithful + cm.both_unfaithful) / n

    # Marginal probabilities
    p_our_faithful = (cm.both_faithful + cm.our_only) / n
    p_our_unfaithful = (cm.both_unfaithful + cm.sonnet_only) / n
    p_son_faithful = (cm.both_faithful + cm.sonnet_only) / n
    p_son_unfaithful = (cm.both_unfaithful + cm.our_only) / n

    p_e = (p_our_faithful * p_son_faithful) + (p_our_unfaithful * p_son_unfaithful)

    if abs(1 - p_e) < 1e-12:
        return float("nan")

    return (p_o - p_e) / (1 - p_e)


def spearman_rho(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Spearman's rank correlation and a two-tailed p-value.

    Returns (rho, p_value). If n < 3, p-value is NaN.
    """
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")

    def rank(vals: list[float]) -> list[float]:
        """Average-rank ties."""
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg = (i + j) / 2.0 + 1  # 1-based average rank
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)

    # Pearson correlation on ranks = Spearman's rho
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x < 1e-12 or den_y < 1e-12:
        return float("nan"), float("nan")

    rho = num / (den_x * den_y)

    # Approximate t-statistic for p-value (Student's t, df = n-2)
    if n < 3 or abs(rho) >= 1.0:
        return rho, float("nan")

    t_stat = rho * math.sqrt(n - 2) / math.sqrt(1 - rho ** 2)
    # Two-tailed p via incomplete beta function approximation (Abramowitz & Stegun)
    p_value = _t_pvalue(t_stat, df=n - 2)

    return rho, p_value


def _t_pvalue(t: float, df: int) -> float:
    """Two-tailed p-value for t-distribution using incomplete beta function.

    Uses the regularized incomplete beta function I_x(a, b) with
    x = df / (df + t^2), a = df/2, b = 0.5.
    Approximation via continued fraction (accurate to ~1e-6).
    """
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5

    # Regularized incomplete beta via continued fraction (Lentz's method)
    def betacf(a: float, b: float, x: float) -> float:
        max_iter = 200
        eps = 3.0e-7
        fpmin = 1.0e-30
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    def log_beta(a: float, b: float) -> float:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    if x < 0.0 or x > 1.0:
        return float("nan")

    if x == 0.0 or x == 1.0:
        ibeta = x
    elif x < (a + 1.0) / (a + b + 2.0):
        ibeta = math.exp(
            a * math.log(x) + b * math.log(1.0 - x) - log_beta(a, b)
        ) * betacf(a, b, x) / a
    else:
        ibeta = 1.0 - math.exp(
            b * math.log(1.0 - x) + a * math.log(x) - log_beta(a, b)
        ) * betacf(b, a, 1.0 - x) / b

    # Two-tailed: ibeta gives P(T <= |t|), so p = ibeta (it's already cumulative
    # from the t-distribution relationship: I_x(df/2, 0.5) = P(|T| > |t|) for
    # the two-tailed test). Return ibeta directly as the two-tailed p-value.
    return ibeta


# ---------------------------------------------------------------------------
# Table printing helpers
# ---------------------------------------------------------------------------

def print_separator(width: int = 80) -> None:
    print("-" * width)


def print_header(title: str, width: int = 80) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def fmt(v: float, decimals: int = 4) -> str:
    if math.isnan(v):
        return "   NaN"
    return f"{v:.{decimals}f}"


def print_table(headers: list[str], rows: list[list[str]], col_widths: list[int]) -> None:
    """Print a simple fixed-width table."""
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print_separator(len(header_line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


# ---------------------------------------------------------------------------
# Analysis 1: Per-case confusion matrices (aggregated by hint type)
# ---------------------------------------------------------------------------

def build_confusion_matrices() -> dict[str, ConfusionCounts]:
    """Load all classified_sonnet JSONL files and tally 2x2 confusion matrices.

    Returns dict mapping hint_type -> ConfusionCounts (aggregated across models).
    Also returns a flat list of all records for downstream use.
    """
    # hint_type -> [both_faithful, our_only, sonnet_only, both_unfaithful]
    counts: dict[str, list[int]] = {h: [0, 0, 0, 0] for h in SONNET_HINT_TYPES}

    for model in MODELS:
        for hint_type in SONNET_HINT_TYPES:
            path = CLASSIFIED_SONNET_DIR / model / f"{hint_type}.jsonl"
            records = load_jsonl(path)
            for rec in records:
                our = rec.get("our_verdict")
                son = rec.get("sonnet_verdict")
                if our is None or son is None:
                    continue
                our_bool = bool(our)
                son_bool = bool(son)
                if our_bool and son_bool:
                    counts[hint_type][0] += 1
                elif our_bool and not son_bool:
                    counts[hint_type][1] += 1
                elif not our_bool and son_bool:
                    counts[hint_type][2] += 1
                else:
                    counts[hint_type][3] += 1

    result = {}
    for hint_type, c in counts.items():
        total = sum(c)
        result[hint_type] = ConfusionCounts(
            both_faithful=c[0],
            our_only=c[1],
            sonnet_only=c[2],
            both_unfaithful=c[3],
            total=total,
        )
    return result


def print_confusion_matrices(cms: dict[str, ConfusionCounts]) -> list[dict]:
    """Print and return CSV rows for confusion matrices."""
    print_header("ANALYSIS 1: Confusion Matrices by Hint Type (Our Pipeline vs Sonnet Judge)")
    print("  Rows = our pipeline verdict, Columns = Sonnet verdict")
    print("  Counts aggregated across all models.\n")

    csv_rows = []
    for hint_type in SONNET_HINT_TYPES:
        cm = cms[hint_type]
        agree = cm.both_faithful + cm.both_unfaithful
        agreement_rate = agree / cm.total if cm.total > 0 else float("nan")

        print(f"  Hint type: {hint_type}  (n={cm.total})")
        print(f"  {'':20s}  Sonnet=FAITHFUL  Sonnet=UNFAITHFUL")
        print(f"  {'Our=FAITHFUL':20s}  {cm.both_faithful:>15d}  {cm.our_only:>17d}")
        print(f"  {'Our=UNFAITHFUL':20s}  {cm.sonnet_only:>15d}  {cm.both_unfaithful:>17d}")
        print(f"  Agreement: {agree}/{cm.total} = {agreement_rate:.4f}")
        print()

        csv_rows.append({
            "hint_type": hint_type,
            "total": cm.total,
            "both_faithful": cm.both_faithful,
            "our_only_faithful": cm.our_only,
            "sonnet_only_faithful": cm.sonnet_only,
            "both_unfaithful": cm.both_unfaithful,
            "agreement_count": agree,
            "agreement_rate": f"{agreement_rate:.4f}",
        })

    return csv_rows


# ---------------------------------------------------------------------------
# Analysis 2: Cohen's kappa per hint type
# ---------------------------------------------------------------------------

def compute_kappas(cms: dict[str, ConfusionCounts]) -> list[dict]:
    """Compute Cohen's kappa for each hint type and return CSV rows."""
    print_header("ANALYSIS 2: Cohen's Kappa Per Hint Type")
    print("  Measures inter-rater agreement beyond chance.")
    print("  kappa < 0.20 = slight, 0.21-0.40 = fair, 0.41-0.60 = moderate,")
    print("  0.61-0.80 = substantial, > 0.80 = almost perfect\n")

    headers = ["hint_type", "n", "agreement_rate", "kappa", "interpretation"]
    col_widths = [16, 6, 16, 8, 20]
    rows = []
    csv_rows = []

    for hint_type in SONNET_HINT_TYPES:
        cm = cms[hint_type]
        kappa = cohens_kappa(cm)
        agreement_rate = (cm.both_faithful + cm.both_unfaithful) / cm.total if cm.total > 0 else float("nan")

        if math.isnan(kappa):
            interp = "N/A"
        elif kappa < 0.20:
            interp = "slight"
        elif kappa < 0.40:
            interp = "fair"
        elif kappa < 0.60:
            interp = "moderate"
        elif kappa < 0.80:
            interp = "substantial"
        else:
            interp = "almost perfect"

        rows.append([hint_type, cm.total, fmt(agreement_rate), fmt(kappa), interp])
        csv_rows.append({
            "hint_type": hint_type,
            "n": cm.total,
            "agreement_rate": fmt(agreement_rate),
            "cohens_kappa": fmt(kappa),
            "interpretation": interp,
        })

    print_table(headers, rows, col_widths)
    print()
    return csv_rows


# ---------------------------------------------------------------------------
# Analysis 3: Regex-only baseline
# ---------------------------------------------------------------------------

def compute_regex_baseline() -> list[dict]:
    """Compute faithfulness rates using regex only (no LLM judge).

    Source: results/analysis/classification_method_breakdown.csv
    regex_faithful / total = regex-only faithfulness rate.
    Compared against full pipeline rate from faithfulness_summary.csv.
    """
    print_header("ANALYSIS 3: Regex-Only Baseline Faithfulness Rates")
    print("  What would faithfulness rates look like with no LLM judge?")
    print("  regex_only_rate = regex_faithful / total_influenced\n")

    breakdown_rows = read_csv_as_dicts(ANALYSIS_DIR / "classification_method_breakdown.csv")
    summary_rows = read_csv_as_dicts(ANALYSIS_DIR / "faithfulness_summary.csv")

    # Build full pipeline rate lookup: model -> rate
    pipeline_rate: dict[str, float] = {}
    for row in summary_rows:
        model = row["model"]
        rate_str = row.get("faithfulness_rate", "")
        try:
            pipeline_rate[model] = float(rate_str)
        except (ValueError, KeyError):
            pass

    # Aggregate regex counts per model across all hint types
    model_regex_faithful: dict[str, int] = defaultdict(int)
    model_total: dict[str, int] = defaultdict(int)

    for row in breakdown_rows:
        model = row["model"]
        try:
            rf = int(row["regex_faithful"])
            total = int(row["total"])
        except (ValueError, KeyError):
            continue
        model_regex_faithful[model] += rf
        model_total[model] += total

    headers = ["model", "total", "regex_faithful", "regex_only_rate", "pipeline_rate", "delta"]
    col_widths = [26, 7, 14, 16, 14, 8]
    rows = []
    csv_rows = []

    for model in MODELS:
        total = model_total.get(model, 0)
        rf = model_regex_faithful.get(model, 0)
        regex_rate = rf / total if total > 0 else float("nan")
        pipe_rate = pipeline_rate.get(model, float("nan"))
        delta = pipe_rate - regex_rate if not (math.isnan(pipe_rate) or math.isnan(regex_rate)) else float("nan")

        rows.append([model, total, rf, fmt(regex_rate), fmt(pipe_rate), fmt(delta)])
        csv_rows.append({
            "model": model,
            "total_influenced": total,
            "regex_faithful": rf,
            "regex_only_rate": fmt(regex_rate),
            "full_pipeline_rate": fmt(pipe_rate),
            "delta_pipeline_minus_regex": fmt(delta),
        })

    print_table(headers, rows, col_widths)
    print()
    return csv_rows


# ---------------------------------------------------------------------------
# Analysis 4: Spearman rank correlation (model rankings)
# ---------------------------------------------------------------------------

def compute_rank_correlation() -> list[dict]:
    """Rank models by each classifier's faithfulness rate and compute Spearman's rho.

    Also identifies models where rankings shift most between classifiers.
    """
    print_header("ANALYSIS 4: Spearman Rank Correlation (Model Ranking Stability)")
    print("  Do both classifiers agree on which models are most/least faithful?")
    print("  Source: sonnet_faithfulness_summary.csv vs faithfulness_summary.csv\n")

    sonnet_rows = read_csv_as_dicts(ANALYSIS_DIR / "sonnet_faithfulness_summary.csv")
    pipeline_rows = read_csv_as_dicts(ANALYSIS_DIR / "faithfulness_summary.csv")

    # Build per-model rate dicts
    sonnet_rate: dict[str, float] = {}
    for row in sonnet_rows:
        model = row["model"]
        try:
            sonnet_rate[model] = float(row["sonnet_rate"])
        except (ValueError, KeyError):
            pass

    pipeline_rate: dict[str, float] = {}
    for row in pipeline_rows:
        model = row["model"]
        try:
            pipeline_rate[model] = float(row["faithfulness_rate"])
        except (ValueError, KeyError):
            pass

    # Only include models present in both
    common_models = [m for m in MODELS if m in sonnet_rate and m in pipeline_rate]

    x = [sonnet_rate[m] for m in common_models]
    y = [pipeline_rate[m] for m in common_models]

    rho, p_value = spearman_rho(x, y)

    print(f"  Models compared: {len(common_models)}")
    print(f"  Spearman's rho:  {fmt(rho)}")
    print(f"  p-value:         {fmt(p_value)}")
    print()

    # Compute individual ranks and rank differences
    def rank_list(vals: list[float]) -> list[int]:
        """Return ranks (1 = highest rate) for each value."""
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        ranks = [0] * len(vals)
        for rank, idx in enumerate(order, start=1):
            ranks[idx] = rank
        return ranks

    ranks_sonnet = rank_list(x)
    ranks_pipeline = rank_list(y)

    headers = ["model", "sonnet_rate", "pipeline_rate", "rank_sonnet", "rank_pipeline", "rank_delta"]
    col_widths = [26, 11, 13, 12, 14, 11]
    rows = []
    csv_rows = []

    model_rows = list(zip(common_models, x, y, ranks_sonnet, ranks_pipeline))
    model_rows_sorted = sorted(model_rows, key=lambda t: t[3])  # sort by sonnet rank

    for model, sr, pr, rs, rp in model_rows_sorted:
        delta = rp - rs
        rows.append([model, fmt(sr), fmt(pr), rs, rp, delta])
        csv_rows.append({
            "model": model,
            "sonnet_rate": fmt(sr),
            "pipeline_rate": fmt(pr),
            "rank_sonnet": rs,
            "rank_pipeline": rp,
            "rank_delta": delta,
        })

    print_table(headers, rows, col_widths)
    print()

    # Highlight biggest movers
    movers = sorted(csv_rows, key=lambda r: abs(int(r["rank_delta"])), reverse=True)
    print("  Biggest rank shifts (|rank_pipeline - rank_sonnet|):")
    for r in movers[:5]:
        print(f"    {r['model']:26s}  sonnet_rank={r['rank_sonnet']:2d}  pipeline_rank={r['rank_pipeline']:2d}  delta={r['rank_delta']:+d}")
    print()

    # Add summary row at top
    summary_row = {
        "model": "SUMMARY",
        "sonnet_rate": fmt(rho),
        "pipeline_rate": "Spearman_rho",
        "rank_sonnet": "",
        "rank_pipeline": "",
        "rank_delta": f"p={fmt(p_value)}",
    }
    return [summary_row] + csv_rows


# ---------------------------------------------------------------------------
# Analysis 5: Overall agreement statistics
# ---------------------------------------------------------------------------

def compute_overall_agreement(cms: dict[str, ConfusionCounts]) -> list[dict]:
    """Compute overall and per-hint-type agreement rates."""
    print_header("ANALYSIS 5: Overall Agreement Statistics")

    total_all = sum(cm.total for cm in cms.values())
    agree_all = sum(cm.both_faithful + cm.both_unfaithful for cm in cms.values())
    faithful_all = sum(cm.both_faithful + cm.our_only for cm in cms.values())
    son_faithful_all = sum(cm.both_faithful + cm.sonnet_only for cm in cms.values())

    overall_agreement = agree_all / total_all if total_all > 0 else float("nan")
    overall_our_rate = faithful_all / total_all if total_all > 0 else float("nan")
    overall_son_rate = son_faithful_all / total_all if total_all > 0 else float("nan")

    print(f"  Total cases compared (across all models x hint types): {total_all}")
    print(f"  Cases where both classifiers agree:                    {agree_all}  ({fmt(overall_agreement)} agreement rate)")
    print(f"  Our pipeline faithfulness rate (overall):              {fmt(overall_our_rate)}")
    print(f"  Sonnet faithfulness rate (overall):                    {fmt(overall_son_rate)}")
    print()

    headers = ["hint_type", "n", "both_faithful", "our_only", "sonnet_only", "both_unf.", "agreement"]
    col_widths = [16, 6, 13, 9, 11, 10, 10]
    rows = []
    csv_rows = []

    # Per hint type
    for hint_type in SONNET_HINT_TYPES:
        cm = cms[hint_type]
        agree = cm.both_faithful + cm.both_unfaithful
        agreement_rate = agree / cm.total if cm.total > 0 else float("nan")
        rows.append([
            hint_type,
            cm.total,
            cm.both_faithful,
            cm.our_only,
            cm.sonnet_only,
            cm.both_unfaithful,
            fmt(agreement_rate),
        ])
        csv_rows.append({
            "hint_type": hint_type,
            "n": cm.total,
            "both_faithful": cm.both_faithful,
            "our_only_faithful": cm.our_only,
            "sonnet_only_faithful": cm.sonnet_only,
            "both_unfaithful": cm.both_unfaithful,
            "agreement_rate": fmt(agreement_rate),
            "our_faithfulness_rate": fmt((cm.both_faithful + cm.our_only) / cm.total if cm.total > 0 else float("nan")),
            "sonnet_faithfulness_rate": fmt((cm.both_faithful + cm.sonnet_only) / cm.total if cm.total > 0 else float("nan")),
        })

    # Overall row
    rows.append([
        "OVERALL",
        total_all,
        agree_all,
        "-",
        "-",
        "-",
        fmt(overall_agreement),
    ])
    csv_rows.append({
        "hint_type": "OVERALL",
        "n": total_all,
        "both_faithful": agree_all,
        "our_only_faithful": "-",
        "sonnet_only_faithful": "-",
        "both_unfaithful": "-",
        "agreement_rate": fmt(overall_agreement),
        "our_faithfulness_rate": fmt(overall_our_rate),
        "sonnet_faithfulness_rate": fmt(overall_son_rate),
    })

    print_table(headers, rows, col_widths)
    print()

    # Disagreement breakdown
    our_only_total = sum(cm.our_only for cm in cms.values())
    son_only_total = sum(cm.sonnet_only for cm in cms.values())
    print(f"  Disagree breakdown:")
    print(f"    Our=faithful, Sonnet=unfaithful: {our_only_total}  ({fmt(our_only_total/total_all)} of cases)")
    print(f"    Our=unfaithful, Sonnet=faithful: {son_only_total}  ({fmt(son_only_total/total_all)} of cases)")
    print(f"  -> Our pipeline labels MORE cases faithful than Sonnet by {our_only_total - son_only_total} net cases")
    print()

    return csv_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("Classifier Comparison Analysis — Paper 2")
    print("'Measuring Faithfulness Depends on How You Measure'")
    print(f"Output directory: {PAPER2_DIR.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    # Step 1: Build confusion matrices (needed by analyses 1, 2, 5)
    # ------------------------------------------------------------------
    print("\nLoading per-case data from classified_sonnet/...")
    cms = build_confusion_matrices()
    loaded = sum(cm.total for cm in cms.values())
    print(f"  Total cases loaded: {loaded}")

    # ------------------------------------------------------------------
    # Analysis 1: Print confusion matrices + save CSV
    # ------------------------------------------------------------------
    cm_csv_rows = print_confusion_matrices(cms)
    write_csv(
        PAPER2_DIR / "confusion_matrix_by_hint.csv",
        cm_csv_rows,
        fieldnames=[
            "hint_type", "total", "both_faithful", "our_only_faithful",
            "sonnet_only_faithful", "both_unfaithful", "agreement_count", "agreement_rate",
        ],
    )

    # ------------------------------------------------------------------
    # Analysis 2: Cohen's kappa + save CSV
    # ------------------------------------------------------------------
    kappa_csv_rows = compute_kappas(cms)
    write_csv(
        PAPER2_DIR / "cohens_kappa_by_hint.csv",
        kappa_csv_rows,
        fieldnames=["hint_type", "n", "agreement_rate", "cohens_kappa", "interpretation"],
    )

    # ------------------------------------------------------------------
    # Analysis 3: Regex-only baseline + save CSV
    # ------------------------------------------------------------------
    regex_csv_rows = compute_regex_baseline()
    write_csv(
        PAPER2_DIR / "regex_only_baseline.csv",
        regex_csv_rows,
        fieldnames=[
            "model", "total_influenced", "regex_faithful",
            "regex_only_rate", "full_pipeline_rate", "delta_pipeline_minus_regex",
        ],
    )

    # ------------------------------------------------------------------
    # Analysis 4: Spearman rank correlation + save CSV
    # ------------------------------------------------------------------
    rank_csv_rows = compute_rank_correlation()
    write_csv(
        PAPER2_DIR / "rank_correlation.csv",
        rank_csv_rows,
        fieldnames=[
            "model", "sonnet_rate", "pipeline_rate",
            "rank_sonnet", "rank_pipeline", "rank_delta",
        ],
    )

    # ------------------------------------------------------------------
    # Analysis 5: Overall agreement + save CSV
    # ------------------------------------------------------------------
    agreement_csv_rows = compute_overall_agreement(cms)
    write_csv(
        PAPER2_DIR / "overall_agreement.csv",
        agreement_csv_rows,
        fieldnames=[
            "hint_type", "n", "both_faithful", "our_only_faithful",
            "sonnet_only_faithful", "both_unfaithful", "agreement_rate",
            "our_faithfulness_rate", "sonnet_faithfulness_rate",
        ],
    )

    print("Done.")
    print()


if __name__ == "__main__":
    main()
