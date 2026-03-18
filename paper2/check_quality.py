#!/usr/bin/env python3
"""Verify ALL mathematical formulas, computed values, and statistical claims in Paper 2."""

import math
from scipy import stats
import numpy as np

print("=" * 80)
print("PAPER 2 VERIFICATION SCRIPT")
print("=" * 80)

errors = []
warnings = []

# ============================================================
# TABLE 1: Model x Classifier
# ============================================================
print("\n" + "=" * 80)
print("TABLE 1: Model x Classifier")
print("=" * 80)

table1 = {
    "DeepSeek-V3.2-Speciale": {"N": 899, "Regex": 94.4, "Pipeline": 97.6, "Sonnet": 89.9, "Delta": 7.7},
    "GPT-OSS-120B": {"N": 769, "Regex": 78.4, "Pipeline": 94.7, "Sonnet": 84.9, "Delta": 9.8},
    "OLMo-3.1-32B": {"N": 997, "Regex": 66.5, "Pipeline": 71.9, "Sonnet": 81.0, "Delta": -9.1},
    "Step-3.5-Flash": {"N": 750, "Regex": 93.2, "Pipeline": 96.0, "Sonnet": 75.3, "Delta": 20.7},
    "DeepSeek-R1": {"N": 1193, "Regex": 91.6, "Pipeline": 94.8, "Sonnet": 74.8, "Delta": 20.0},
    "MiniMax-M2.5": {"N": 554, "Regex": 85.5, "Pipeline": 91.2, "Sonnet": 73.1, "Delta": 18.1},
    "Qwen3.5-27B": {"N": 1308, "Regex": 97.5, "Pipeline": 98.9, "Sonnet": 68.3, "Delta": 30.6},
    "ERNIE-4.5-21B": {"N": 900, "Regex": 67.6, "Pipeline": 75.1, "Sonnet": 62.8, "Delta": 12.3},
    "Nemotron-Nano-9B": {"N": 732, "Regex": 37.5, "Pipeline": 67.4, "Sonnet": 60.9, "Delta": 6.5},
    "OLMo-3-7B": {"N": 580, "Regex": 70.9, "Pipeline": 80.2, "Sonnet": 56.9, "Delta": 23.3},
    "QwQ-32B": {"N": 982, "Regex": 55.1, "Pipeline": 66.5, "Sonnet": 56.3, "Delta": 10.2},
    "Seed-1.6-Flash": {"N": 612, "Regex": 26.5, "Pipeline": 37.1, "Sonnet": 39.7, "Delta": -2.6},
}

# Check 1: Delta = Pipeline - Sonnet for every row
print("\nCheck 1: Delta = Pipeline - Sonnet")
for model, vals in table1.items():
    computed_delta = round(vals["Pipeline"] - vals["Sonnet"], 1)
    if computed_delta != vals["Delta"]:
        msg = f"  DISCREPANCY: {model}: Delta claimed={vals['Delta']}, computed={computed_delta}"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: {model}: Delta={vals['Delta']}")

# Check 2: Total N sums to 10276
total_n = sum(v["N"] for v in table1.values())
print(f"\nCheck 2: Total N = {total_n} (claimed 10,276)")
if total_n != 10276:
    msg = f"  DISCREPANCY: Total N={total_n}, claimed=10276"
    print(msg)
    errors.append(msg)
else:
    print("  OK")

# Check 3: Overall row as weighted averages
print("\nCheck 3: Overall row as weighted averages")
for col in ["Regex", "Pipeline", "Sonnet"]:
    weighted = sum(v[col] * v["N"] / 100 for v in table1.values())
    weighted_pct = weighted / total_n * 100
    print(f"  {col}: computed={weighted_pct:.1f}%")

claimed_overall = {"Regex": 74.4, "Pipeline": 82.6, "Sonnet": 69.7}
for col, claimed in claimed_overall.items():
    weighted = sum(v[col] * v["N"] / 100 for v in table1.values())
    weighted_pct = weighted / total_n * 100
    diff = abs(weighted_pct - claimed)
    if diff > 0.15:
        msg = f"  DISCREPANCY: {col} Overall: claimed={claimed}, computed={weighted_pct:.2f}, diff={diff:.2f}pp"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: {col} Overall: claimed={claimed}, computed={weighted_pct:.2f}")

# Check 4: Overall Delta
overall_delta = round(claimed_overall["Pipeline"] - claimed_overall["Sonnet"], 1)
claimed_overall_delta = 12.9
print(f"\nCheck 4: Overall Delta: computed={overall_delta}, claimed={claimed_overall_delta}")
if overall_delta != claimed_overall_delta:
    msg = f"  DISCREPANCY: Overall Delta: claimed={claimed_overall_delta}, computed={overall_delta}"
    print(msg)
    errors.append(msg)
else:
    print("  OK")

# Check 5: Table 1 sorted by Sonnet rate (descending)
print("\nCheck 5: Models sorted by Sonnet rate (descending)")
models_sorted = sorted(table1.items(), key=lambda x: x[1]["Sonnet"], reverse=True)
sonnet_order = [m[0] for m in models_sorted]
table_order = list(table1.keys())
if sonnet_order != table_order:
    msg = f"  DISCREPANCY: Table not sorted by Sonnet rate"
    print(msg)
    for i, (expected, actual) in enumerate(zip(sonnet_order, table_order)):
        if expected != actual:
            print(f"    Position {i+1}: expected {expected} ({table1[expected]['Sonnet']}), got {actual} ({table1[actual]['Sonnet']})")
    errors.append(msg)
else:
    print("  OK: Models correctly sorted by Sonnet rate descending")

# ============================================================
# TABLE 2: Hint x Classifier
# ============================================================
print("\n" + "=" * 80)
print("TABLE 2: Hint x Classifier")
print("=" * 80)

table2 = {
    "Sycophancy":  {"Regex": 96.8, "Pipeline": 97.3, "Sonnet": 53.9, "Delta": 43.4},
    "Consistency": {"Regex": 67.9, "Pipeline": 68.6, "Sonnet": 35.5, "Delta": 33.1},
    "Metadata":    {"Regex": 59.8, "Pipeline": 60.4, "Sonnet": 69.9, "Delta": -9.5},
    "Unethical":   {"Regex": 87.7, "Pipeline": 88.4, "Sonnet": 79.4, "Delta": 9.0},
    "Grader":      {"Regex": 79.8, "Pipeline": 80.6, "Sonnet": 77.7, "Delta": 2.9},
}

# Check Delta = Pipeline - Sonnet
print("\nCheck: Delta = Pipeline - Sonnet")
for hint, vals in table2.items():
    computed_delta = round(vals["Pipeline"] - vals["Sonnet"], 1)
    if computed_delta != vals["Delta"]:
        msg = f"  DISCREPANCY: {hint}: Delta claimed={vals['Delta']}, computed={computed_delta}"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: {hint}: Delta={vals['Delta']}")

# Check Table 2 sorted by |Delta| (descending)
print("\nCheck: Sorted by |Delta| descending")
sorted_by_delta = sorted(table2.items(), key=lambda x: abs(x[1]["Delta"]), reverse=True)
sorted_order = [h[0] for h in sorted_by_delta]
table2_order = list(table2.keys())
if sorted_order != table2_order:
    msg = f"  DISCREPANCY: Table 2 not sorted by |Delta|. Expected: {sorted_order}, Got: {table2_order}"
    print(msg)
    errors.append(msg)
else:
    print("  OK")

# ============================================================
# TABLE 3: Confusion Matrix
# ============================================================
print("\n" + "=" * 80)
print("TABLE 3: Confusion Matrix")
print("=" * 80)

table3 = {
    "Sycophancy":  {"Both_F": 1095, "Pipe_only": 883, "Son_only": 2,   "Both_U": 54,  "N": 2034},
    "Consistency": {"Both_F": 179,  "Pipe_only": 267, "Son_only": 52,  "Both_U": 152, "N": 650},
    "Metadata":    {"Both_F": 773,  "Pipe_only": 151, "Son_only": 297, "Both_U": 310, "N": 1531},
    "Grader":      {"Both_F": 1968, "Pipe_only": 313, "Son_only": 230, "Both_U": 318, "N": 2829},
    "Unethical":   {"Both_F": 2424, "Pipe_only": 432, "Son_only": 141, "Both_U": 235, "N": 3232},
}

# Check 1: Row sums = N
print("\nCheck 1: Both_F + Pipe_only + Son_only + Both_U = N")
for hint, vals in table3.items():
    row_sum = vals["Both_F"] + vals["Pipe_only"] + vals["Son_only"] + vals["Both_U"]
    if row_sum != vals["N"]:
        msg = f"  DISCREPANCY: {hint}: sum={row_sum}, N={vals['N']}"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: {hint}: sum={row_sum} = N={vals['N']}")

# Check 2: Total N across hints
total_hint_n = sum(v["N"] for v in table3.values())
print(f"\nCheck 2: Total across hints = {total_hint_n} (should be 10,276)")
if total_hint_n != 10276:
    msg = f"  DISCREPANCY: Total hint N={total_hint_n}, expected 10276"
    print(msg)
    errors.append(msg)
else:
    print("  OK")

# Check 3: Confusion matrix implies correct faithfulness rates in Table 2
print("\nCheck 3: Confusion matrix -> faithfulness rates (Table 2)")
for hint, cm in table3.items():
    N = cm["N"]
    pipeline_faithful = (cm["Both_F"] + cm["Pipe_only"]) / N * 100
    sonnet_faithful = (cm["Both_F"] + cm["Son_only"]) / N * 100

    t2 = table2[hint]
    pipe_diff = abs(pipeline_faithful - t2["Pipeline"])
    son_diff = abs(sonnet_faithful - t2["Sonnet"])

    if pipe_diff > 0.15:
        msg = f"  DISCREPANCY: {hint} Pipeline: Table2={t2['Pipeline']}, from CM={pipeline_faithful:.1f}"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: {hint} Pipeline: Table2={t2['Pipeline']}, from CM={pipeline_faithful:.1f}")

    if son_diff > 0.15:
        msg = f"  DISCREPANCY: {hint} Sonnet: Table2={t2['Sonnet']}, from CM={sonnet_faithful:.1f}"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: {hint} Sonnet: Table2={t2['Sonnet']}, from CM={sonnet_faithful:.1f}")


# ============================================================
# TABLE 4: Kappa and McNemar
# ============================================================
print("\n" + "=" * 80)
print("TABLE 4: Kappa and McNemar verification from confusion matrix")
print("=" * 80)

table4_claimed = {
    "Sycophancy":  {"N": 2034,  "Agree": 56.5, "Kappa": 0.06, "Chi2": 877.0},
    "Consistency": {"N": 650,   "Agree": 50.9, "Kappa": 0.11, "Chi2": 144.9},
    "Metadata":    {"N": 1531,  "Agree": 70.7, "Kappa": 0.36, "Chi2": 47.6},
    "Grader":      {"N": 2829,  "Agree": 80.8, "Kappa": 0.42, "Chi2": 12.7},
    "Unethical":   {"N": 3232,  "Agree": 82.3, "Kappa": 0.35, "Chi2": 147.8},
}

for hint in table3:
    cm = table3[hint]
    claimed = table4_claimed[hint]
    N = cm["N"]
    a, b, c, d = cm["Both_F"], cm["Pipe_only"], cm["Son_only"], cm["Both_U"]

    # Agreement = (Both_F + Both_U) / N
    p_o = (a + d) / N
    agree_pct = p_o * 100
    print(f"\n--- {hint} (N={N}) ---")
    agree_diff = abs(agree_pct - claimed["Agree"])
    if agree_diff > 0.15:
        msg = f"  DISCREPANCY: {hint} Agreement: claimed={claimed['Agree']}, computed={agree_pct:.1f}"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: Agreement: claimed={claimed['Agree']}, computed={agree_pct:.1f}")

    # p_e = p_f1 * p_f2 + (1-p_f1)*(1-p_f2)
    p_f1 = (a + b) / N  # Pipeline faithful rate
    p_f2 = (a + c) / N  # Sonnet faithful rate
    p_e = p_f1 * p_f2 + (1 - p_f1) * (1 - p_f2)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0.0

    kappa_diff = abs(kappa - claimed["Kappa"])
    if kappa_diff > 0.015:
        msg = f"  DISCREPANCY: {hint} Kappa: claimed={claimed['Kappa']}, computed={kappa:.3f} (p_o={p_o:.4f}, p_e={p_e:.4f})"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: Kappa: claimed={claimed['Kappa']}, computed={kappa:.3f} (p_o={p_o:.4f}, p_e={p_e:.4f})")

    # McNemar chi-squared = (b - c)^2 / (b + c)
    if (b + c) > 0:
        chi2 = (b - c) ** 2 / (b + c)
    else:
        chi2 = 0.0
    chi2_diff = abs(chi2 - claimed["Chi2"])
    if chi2_diff > 0.15:
        msg = f"  DISCREPANCY: {hint} McNemar chi2: claimed={claimed['Chi2']}, computed={chi2:.1f}"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: McNemar chi2: claimed={claimed['Chi2']}, computed={chi2:.1f}")

    # Check p < 0.001
    p_val = 1 - stats.chi2.cdf(chi2, df=1)
    if p_val >= 0.001:
        msg = f"  DISCREPANCY: {hint} McNemar p={p_val:.6f}, claimed p < 0.001"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: McNemar p={p_val:.2e} < 0.001")

# Overall kappa from total confusion matrix
print("\n--- Overall ---")
total_a = sum(table3[h]["Both_F"] for h in table3)
total_b = sum(table3[h]["Pipe_only"] for h in table3)
total_c = sum(table3[h]["Son_only"] for h in table3)
total_d = sum(table3[h]["Both_U"] for h in table3)
total_N = total_a + total_b + total_c + total_d
p_o_total = (total_a + total_d) / total_N
agree_total = p_o_total * 100
p_f1_total = (total_a + total_b) / total_N
p_f2_total = (total_a + total_c) / total_N
p_e_total = p_f1_total * p_f2_total + (1 - p_f1_total) * (1 - p_f2_total)
kappa_total = (p_o_total - p_e_total) / (1 - p_e_total)

print(f"  Total confusion: a={total_a}, b={total_b}, c={total_c}, d={total_d}, N={total_N}")
print(f"  Agreement: claimed=73.1, computed={agree_total:.1f}")
print(f"  Kappa: claimed=0.28, computed={kappa_total:.3f}")

# Overall McNemar
chi2_total = (total_b - total_c) ** 2 / (total_b + total_c)
print(f"  McNemar chi2: claimed=633.3, computed={chi2_total:.1f}")

for label, claimed_val, computed_val in [
    ("Agreement", 73.1, agree_total),
    ("Kappa", 0.28, kappa_total),
    ("McNemar chi2", 633.3, chi2_total),
]:
    tolerance = 0.15 if label != "Kappa" else 0.015
    if abs(computed_val - claimed_val) > tolerance:
        msg = f"  DISCREPANCY: Overall {label}: claimed={claimed_val}, computed={computed_val:.3f}"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: Overall {label}")

# ============================================================
# WILSON SCORE CONFIDENCE INTERVALS
# ============================================================
print("\n" + "=" * 80)
print("WILSON SCORE CONFIDENCE INTERVALS")
print("=" * 80)

n = 10276

def wilson_ci(p_hat, n, z=1.96):
    """Wilson score interval."""
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (center - spread, center + spread)

# Pipeline: claimed 81.9% - 83.3%
p_pipeline = 0.826
lo, hi = wilson_ci(p_pipeline, n)
print(f"Pipeline Wilson CI: claimed [81.9%, 83.3%], computed [{lo*100:.1f}%, {hi*100:.1f}%]")
if abs(lo * 100 - 81.9) > 0.15 or abs(hi * 100 - 83.3) > 0.15:
    msg = f"  DISCREPANCY: Pipeline Wilson CI: claimed [81.9, 83.3], computed [{lo*100:.1f}, {hi*100:.1f}]"
    print(msg)
    errors.append(msg)
else:
    print("  OK")

# Sonnet: claimed 68.8% - 70.6%
p_sonnet = 0.697
lo, hi = wilson_ci(p_sonnet, n)
print(f"Sonnet Wilson CI: claimed [68.8%, 70.6%], computed [{lo*100:.1f}%, {hi*100:.1f}%]")
if abs(lo * 100 - 68.8) > 0.15 or abs(hi * 100 - 70.6) > 0.15:
    msg = f"  DISCREPANCY: Sonnet Wilson CI: claimed [68.8, 70.6], computed [{lo*100:.1f}, {hi*100:.1f}]"
    print(msg)
    errors.append(msg)
else:
    print("  OK")

# ============================================================
# SPEARMAN RANK CORRELATION
# ============================================================
print("\n" + "=" * 80)
print("SPEARMAN RANK CORRELATION")
print("=" * 80)

# Pipeline rates in table order (sorted by Sonnet)
pipeline_rates = [v["Pipeline"] for v in table1.values()]
sonnet_rates = [v["Sonnet"] for v in table1.values()]

rho, p_val = stats.spearmanr(pipeline_rates, sonnet_rates)
print(f"Spearman rho: claimed=0.64, computed={rho:.4f}")
print(f"Spearman p: claimed=0.024, computed={p_val:.4f}")

if abs(rho - 0.64) > 0.015:
    msg = f"  DISCREPANCY: Spearman rho: claimed=0.64, computed={rho:.4f}"
    print(msg)
    errors.append(msg)
else:
    print("  OK: rho")

if abs(p_val - 0.024) > 0.005:
    msg = f"  DISCREPANCY: Spearman p: claimed=0.024, computed={p_val:.4f}"
    print(msg)
    errors.append(msg)
else:
    print("  OK: p-value")

# Fisher z-transformation CI for rho
# Fisher z = 0.5 * ln((1+r)/(1-r))
# SE(z) = 1/sqrt(n-3)
# CI on z, then back-transform
n_models = 12
z_rho = 0.5 * math.log((1 + rho) / (1 - rho))
se_z = 1 / math.sqrt(n_models - 3)
z_lo = z_rho - 1.96 * se_z
z_hi = z_rho + 1.96 * se_z
rho_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
rho_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)
print(f"Fisher z CI on rho: claimed [0.10, 0.89], computed [{rho_lo:.2f}, {rho_hi:.2f}]")

if abs(rho_lo - 0.10) > 0.05 or abs(rho_hi - 0.89) > 0.05:
    msg = f"  DISCREPANCY: Fisher z CI: claimed [0.10, 0.89], computed [{rho_lo:.2f}, {rho_hi:.2f}]"
    print(msg)
    errors.append(msg)
else:
    print("  OK")

# ============================================================
# SYCOPHANCY GAP IS 15 TIMES LARGER THAN GRADER GAP
# ============================================================
print("\n" + "=" * 80)
print("'SYCOPHANCY GAP IS 15 TIMES LARGER THAN GRADER GAP'")
print("=" * 80)

ratio = 43.4 / 2.9
print(f"43.4 / 2.9 = {ratio:.2f}")
if abs(ratio - 15) > 0.5:
    msg = f"  DISCREPANCY: 43.4/2.9 = {ratio:.2f}, claimed '15 times'"
    print(msg)
    errors.append(msg)
else:
    print("  OK: approximately 15x")

# ============================================================
# MODEL RANKINGS
# ============================================================
print("\n" + "=" * 80)
print("MODEL RANKINGS")
print("=" * 80)

# Rank by Pipeline (descending)
pipeline_ranked = sorted(table1.items(), key=lambda x: x[1]["Pipeline"], reverse=True)
pipeline_ranks = {m: i+1 for i, (m, _) in enumerate(pipeline_ranked)}

# Rank by Sonnet (descending)
sonnet_ranked = sorted(table1.items(), key=lambda x: x[1]["Sonnet"], reverse=True)
sonnet_ranks = {m: i+1 for i, (m, _) in enumerate(sonnet_ranked)}

print("\nModel Pipeline-Rank -> Sonnet-Rank")
for m in table1:
    print(f"  {m}: Pipeline={pipeline_ranks[m]}, Sonnet={sonnet_ranks[m]}")

# Claims:
# Qwen3.5-27B: Pipeline=1, Sonnet=7
# OLMo-3.1-32B: Pipeline=9, Sonnet=3  (but text also says "last" by pipeline)
print(f"\nClaimed: Qwen3.5-27B Pipeline=1, Sonnet=7")
print(f"Computed: Qwen3.5-27B Pipeline={pipeline_ranks['Qwen3.5-27B']}, Sonnet={sonnet_ranks['Qwen3.5-27B']}")
if pipeline_ranks["Qwen3.5-27B"] != 1:
    msg = f"  DISCREPANCY: Qwen3.5-27B pipeline rank={pipeline_ranks['Qwen3.5-27B']}, claimed=1"
    print(msg)
    errors.append(msg)
else:
    print("  OK: Pipeline rank 1")
if sonnet_ranks["Qwen3.5-27B"] != 7:
    msg = f"  DISCREPANCY: Qwen3.5-27B sonnet rank={sonnet_ranks['Qwen3.5-27B']}, claimed=7"
    print(msg)
    errors.append(msg)
else:
    print("  OK: Sonnet rank 7")

# OLMo-3.1-32B: Pipeline 9th to Sonnet 3rd (text says "last" for pipeline in sec 4.4, "9th" in intro)
print(f"\nClaimed: OLMo-3.1-32B Pipeline=9 (also 'last'), Sonnet=3")
print(f"Computed: OLMo-3.1-32B Pipeline={pipeline_ranks['OLMo-3.1-32B']}, Sonnet={sonnet_ranks['OLMo-3.1-32B']}")
if pipeline_ranks["OLMo-3.1-32B"] != 9:
    msg = f"  DISCREPANCY: OLMo-3.1-32B pipeline rank={pipeline_ranks['OLMo-3.1-32B']}, claimed=9"
    print(msg)
    errors.append(msg)
else:
    print("  OK: Pipeline rank 9")

# Check "last" claim - is 9 really last out of 12? No, 12th would be last.
if pipeline_ranks["OLMo-3.1-32B"] != 12:
    msg = f"  WARNING: Sec 4.4 says OLMo-3.1-32B 'ranks last by the pipeline' but rank={pipeline_ranks['OLMo-3.1-32B']}/12, not 12th"
    print(msg)
    warnings.append(msg)

if sonnet_ranks["OLMo-3.1-32B"] != 3:
    msg = f"  DISCREPANCY: OLMo-3.1-32B sonnet rank={sonnet_ranks['OLMo-3.1-32B']}, claimed=3"
    print(msg)
    errors.append(msg)
else:
    print("  OK: Sonnet rank 3")

# Fig 4 caption says OLMo-3.1-32B "rises from 9 to 3"
# Intro says "from 9th to 3rd"
# Sec 4.4 says "ranks last by the pipeline (71.9%) but 1st by Sonnet (81.0%)"
# Check: is OLMo-3.1-32B 1st by Sonnet?
if sonnet_ranks["OLMo-3.1-32B"] != 1:
    msg = f"  DISCREPANCY: Sec 4.4 says OLMo-3.1-32B '1st by Sonnet' but rank={sonnet_ranks['OLMo-3.1-32B']}"
    print(msg)
    errors.append(msg)

# ============================================================
# ABSTRACT CLAIMS vs TABLE VALUES
# ============================================================
print("\n" + "=" * 80)
print("ABSTRACT CROSS-CHECKS")
print("=" * 80)

abstract_claims = {
    "Overall Regex": (74.4, claimed_overall["Regex"]),
    "Overall Pipeline": (82.6, claimed_overall["Pipeline"]),
    "Overall Sonnet": (69.7, claimed_overall["Sonnet"]),
    "Per-model gap min (2.6pp)": (2.6, min(abs(v["Delta"]) for v in table1.values())),
    "Per-model gap max (30.6pp)": (30.6, max(abs(v["Delta"]) for v in table1.values())),
    "Kappa sycophancy (0.06)": (0.06, table4_claimed["Sycophancy"]["Kappa"]),
    "Kappa grader (0.42)": (0.42, table4_claimed["Grader"]["Kappa"]),
    "Sycophancy Pipe-only (883)": (883, table3["Sycophancy"]["Pipe_only"]),
    "Sycophancy Son-only (2)": (2, table3["Sycophancy"]["Son_only"]),
    "Sycophancy gap (43.4pp)": (43.4, table2["Sycophancy"]["Delta"]),
    "Grader gap (2.9pp)": (2.9, table2["Grader"]["Delta"]),
}

for claim_name, (abstract_val, table_val) in abstract_claims.items():
    if abstract_val != table_val:
        msg = f"  DISCREPANCY: {claim_name}: abstract={abstract_val}, table={table_val}"
        print(msg)
        errors.append(msg)
    else:
        print(f"  OK: {claim_name}: {abstract_val}")

# Abstract says "Qwen3.5-27B ranks 1st under pipeline but 7th under Sonnet"
# Already checked above

# Abstract: "OLMo-3.1-32B moves in the opposite direction, from 9th to 3rd"
# Already checked above

# ============================================================
# INTRO vs TABLE CROSS-CHECKS
# ============================================================
print("\n" + "=" * 80)
print("INTRODUCTION CROSS-CHECKS")
print("=" * 80)

# Intro: DeepSeek-R1 94.8% pipeline, 74.8% Sonnet
print(f"DeepSeek-R1 Pipeline: intro=94.8, table={table1['DeepSeek-R1']['Pipeline']}")
if table1["DeepSeek-R1"]["Pipeline"] != 94.8:
    msg = f"  DISCREPANCY: DeepSeek-R1 Pipeline"
    errors.append(msg)
else:
    print("  OK")

print(f"DeepSeek-R1 Sonnet: intro=74.8, table={table1['DeepSeek-R1']['Sonnet']}")
if table1["DeepSeek-R1"]["Sonnet"] != 74.8:
    msg = f"  DISCREPANCY: DeepSeek-R1 Sonnet"
    errors.append(msg)
else:
    print("  OK")

# Intro: Qwen3.5-27B 98.9% pipeline, 68.3% Sonnet
print(f"Qwen3.5-27B Pipeline: intro=98.9, table={table1['Qwen3.5-27B']['Pipeline']}")
print(f"Qwen3.5-27B Sonnet: intro=68.3, table={table1['Qwen3.5-27B']['Sonnet']}")

# Intro: consistency gap is mentioned in bullet 2 but not in text discussion
# Discussion: "43.4-percentage-point range for sycophancy hints"
print(f"\nDiscussion sycophancy range: claimed=43.4, Table2 Delta={table2['Sycophancy']['Delta']}")

# ============================================================
# SECTION 4.4: OLMo ranking claims
# ============================================================
print("\n" + "=" * 80)
print("SECTION 4.4 RANKING CLAIMS (detailed)")
print("=" * 80)

# Text says: "OLMo-3.1-32B ... ranks last by the pipeline (71.9%) but 1st by Sonnet (81.0%)"
# Pipeline 71.9 matches table. Let's check if it's truly last.
print("OLMo-3.1-32B pipeline rate: 71.9%")
lower_pipeline = [m for m, v in table1.items() if v["Pipeline"] < 71.9]
print(f"  Models with lower pipeline rate: {lower_pipeline}")
# Seed-1.6-Flash=37.1, QwQ-32B=66.5, Nemotron-Nano-9B=67.4 are all lower
# So OLMo-3.1-32B is NOT last by pipeline.

# "1st by Sonnet (81.0%)" - but DeepSeek-V3.2-Speciale has 89.9%
print(f"OLMo-3.1-32B sonnet rate: 81.0%")
higher_sonnet = [m for m, v in table1.items() if v["Sonnet"] > 81.0]
print(f"  Models with higher Sonnet rate: {higher_sonnet}")

# ============================================================
# EQUATION VERIFICATION
# ============================================================
print("\n" + "=" * 80)
print("EQUATION VERIFICATION")
print("=" * 80)

# Eq 1: Faithfulness Rate = |{i : C(i) = faithful}| / |I|
print("Eq 1 (Faithfulness Rate): Standard ratio definition. CORRECT.")

# Eq 2: Agreement = |{i : C1(i) = C2(i)}| / |I|
print("Eq 2 (Raw Agreement): Standard agreement definition. CORRECT.")

# Eq 3: kappa = (p_o - p_e) / (1 - p_e)
print("Eq 3 (Cohen's kappa): Standard formula. CORRECT.")

# Eq 4: p_e = p_f1 * p_f2 + (1 - p_f1)(1 - p_f2)
print("Eq 4 (p_e expansion): Standard expected agreement for 2x2. CORRECT.")
# Verify: for two binary classifiers with marginals p_f1 and p_f2,
# expected agreement = P(both F) + P(both U) = p_f1*p_f2 + (1-p_f1)*(1-p_f2)

# Eq 5: chi^2_McNemar = (b-c)^2 / (b+c)
print("Eq 5 (McNemar's chi-squared): Standard formula. CORRECT.")

# ============================================================
# COST CLAIM
# ============================================================
print("\n" + "=" * 80)
print("COST CLAIMS")
print("=" * 80)
print("Claimed: $48.99 for 10,276 cases (~$0.005/case)")
cost_per_case = 48.99 / 10276
print(f"  $48.99 / 10276 = ${cost_per_case:.4f}/case")
if abs(cost_per_case - 0.005) > 0.001:
    msg = f"  DISCREPANCY: cost per case = ${cost_per_case:.4f}, claimed ~$0.005"
    print(msg)
    errors.append(msg)
else:
    print("  OK: approximately $0.005/case")

# Cost-accuracy paragraph: "74.4% vs 69.7%, a gap of 4.7 percentage points"
gap_regex_sonnet = 74.4 - 69.7
print(f"\nRegex vs Sonnet gap: claimed=4.7pp, computed={gap_regex_sonnet:.1f}pp")
if abs(gap_regex_sonnet - 4.7) > 0.05:
    msg = f"  DISCREPANCY: Regex-Sonnet gap: claimed=4.7, computed={gap_regex_sonnet:.1f}"
    print(msg)
    errors.append(msg)
else:
    print("  OK")

# ============================================================
# CROSS-REFERENCE: "9 families" in abstract, "12 model families" in setup
# ============================================================
print("\n" + "=" * 80)
print("FAMILY COUNT CROSS-CHECK")
print("=" * 80)
# Abstract says "9 families", Setup says "12 model families"
# Count unique families from model names
# Table 1 has 12 models. Let's see what the text says.
print("Abstract says: '9 families'")
print("Setup (03_setup.tex line 13) says: '12 model families'")
# These need checking against the actual models:
# DeepSeek (2 models), OLMo (2 models), Qwen (2 models: Qwen3.5 + QwQ),
# GPT-OSS, Step, MiniMax, ERNIE, Nemotron, Seed
# That's 9 families with 12 models
families = {
    "DeepSeek": ["DeepSeek-V3.2-Speciale", "DeepSeek-R1"],
    "OpenAI": ["GPT-OSS-120B"],
    "AI2/OLMo": ["OLMo-3.1-32B", "OLMo-3-7B"],
    "StepFun": ["Step-3.5-Flash"],
    "MiniMax": ["MiniMax-M2.5"],
    "Qwen": ["Qwen3.5-27B", "QwQ-32B"],
    "Baidu": ["ERNIE-4.5-21B"],
    "NVIDIA": ["Nemotron-Nano-9B"],
    "ByteDance": ["Seed-1.6-Flash"],
}
print(f"Actual families: {len(families)} = {list(families.keys())}")
if len(families) != 9:
    msg = f"  DISCREPANCY: Family count: actual={len(families)}"
    print(msg)
    errors.append(msg)

msg = "  WARNING: Setup text says '12 model families' but abstract says '9 families'. There are 12 models but only 9 families."
print(msg)
warnings.append(msg)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nTotal ERRORS found: {len(errors)}")
for e in errors:
    print(f"  - {e}")
print(f"\nTotal WARNINGS found: {len(warnings)}")
for w in warnings:
    print(f"  - {w}")
