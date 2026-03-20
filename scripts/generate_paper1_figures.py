"""Generate all figures for Paper 1: Lie to Me — CoT Faithfulness Study.

All data is loaded from results/analysis/ CSVs to ensure reproducibility.
No headline numbers are hardcoded; figures update automatically when
the underlying analysis is re-run.
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')
ANALYSIS_DIR = os.path.join(ROOT_DIR, 'results', 'analysis')
OUT_DIR = os.path.join(ROOT_DIR, 'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Clean academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ============================================================
# DATA LOADING from CSVs
# ============================================================

def load_csv(filename):
    """Load a CSV file from the analysis directory and return list of dicts."""
    path = os.path.join(ANALYSIS_DIR, filename)
    if not os.path.exists(path):
        print(f"ERROR: Missing CSV: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


# Display name mapping (API slug -> short name)
DISPLAY_NAMES = {
    'deepseek-v3.2-speciale': ('DeepSeek-V3.2-Speciale', 'DS-V3.2'),
    'gpt-oss-120b': ('GPT-OSS-120B', 'GPT-OSS'),
    'olmo-3.1-32b-think': ('OLMo-3.1-32B', 'OLMo-32B'),
    'step-3.5-flash': ('Step-3.5-Flash', 'Step-3.5'),
    'deepseek-r1': ('DeepSeek-R1', 'DS-R1'),
    'minimax-m2.5': ('MiniMax-M2.5', 'MiniMax'),
    'qwen3.5-27b': ('Qwen3.5-27B', 'Qwen3.5'),
    'ernie-4.5-21b': ('ERNIE-4.5-21B', 'ERNIE-4.5'),
    'nemotron-nano-9b': ('Nemotron-Nano-9B', 'Nemotron'),
    'olmo-3-7b-think': ('OLMo-3-7B', 'OLMo-7B'),
    'qwq-32b': ('QwQ-32B', 'QwQ-32B'),
    'seed-1.6-flash': ('Seed-1.6-Flash', 'Seed-1.6'),
}

# Active parameters (billions) — from model cards, not derivable from CSVs
ACTIVE_PARAMS_MAP = {
    'deepseek-v3.2-speciale': 37,
    'gpt-oss-120b': 5.1,
    'olmo-3.1-32b-think': 32,
    'step-3.5-flash': 11,
    'deepseek-r1': 37,
    'minimax-m2.5': 10,
    'qwen3.5-27b': 27,
    'ernie-4.5-21b': 3,
    'nemotron-nano-9b': 9,
    'olmo-3-7b-think': 7,
    'qwq-32b': 32,
    'seed-1.6-flash': None,  # Undisclosed
}


def get_short(model_slug):
    """Get short display name for a model slug."""
    entry = DISPLAY_NAMES.get(model_slug)
    return entry[1] if entry else model_slug


def get_full(model_slug):
    """Get full display name for a model slug."""
    entry = DISPLAY_NAMES.get(model_slug)
    return entry[0] if entry else model_slug


# Load all data
sonnet_faith = load_csv('sonnet_faithfulness_rate.csv')
pipeline_faith = load_csv('faithfulness_rate.csv')
base_acc = load_csv('base_accuracy.csv')
influence = load_csv('influence_rate.csv')
cot_length = load_csv('cot_length_stats.csv')
thinking_answer = load_csv('thinking_vs_answer_by_model.csv')
sonnet_by_hint = load_csv('sonnet_faithfulness_by_hint.csv')

# Build model-keyed lookups
sonnet_by_model = {r['model']: float(r['avg']) * 100 for r in sonnet_faith}
pipeline_by_model = {r['model']: float(r['avg']) * 100 for r in pipeline_faith}
accuracy_by_model = {r['model']: float(r['accuracy']) * 100 for r in base_acc}
influence_by_model = {r['model']: float(r['avg']) * 100 for r in influence}
cot_by_model = {r['model']: float(r['median_reasoning_tokens']) for r in cot_length}
thinking_by_model = {r['model']: float(r['thinking_rate']) * 100 for r in thinking_answer}
answer_by_model = {r['model']: float(r['answer_rate']) * 100 for r in thinking_answer}

# Sort models by Sonnet faithfulness (descending) — canonical order
MODELS_SORTED = sorted(sonnet_by_model.keys(), key=lambda m: sonnet_by_model[m], reverse=True)

# Hint-type Sonnet rates (from sonnet_faithfulness_by_hint.csv)
HINT_TYPES = ['consistency', 'sycophancy', 'metadata', 'grader', 'unethical']
HINT_LABELS = ['Consistency', 'Sycophancy', 'Metadata', 'Grader', 'Unethical']
hint_sonnet_rates = {}
hint_ns = {}
for row in sonnet_by_hint:
    hint_sonnet_rates[row['hint_type']] = float(row['sonnet_rate']) * 100
    hint_ns[row['hint_type']] = int(row['total'])

# Per-model per-hint Sonnet faithfulness (from sonnet_faithfulness_rate.csv)
sonnet_per_hint = {}
for row in sonnet_faith:
    model = row['model']
    sonnet_per_hint[model] = {ht: float(row[ht]) * 100 for ht in HINT_TYPES}

# Per-model per-hint influence rates (from influence_rate.csv)
INFL_HINT_TYPES = ['sycophancy', 'consistency', 'visual_pattern', 'metadata', 'grader', 'unethical']
INFL_HINT_LABELS = ['Sycophancy', 'Consistency', 'Visual\nPattern', 'Metadata', 'Grader', 'Unethical']
influence_per_hint = {}
for row in influence:
    model = row['model']
    influence_per_hint[model] = {ht: float(row[ht]) * 100 for ht in INFL_HINT_TYPES}

# Overall Sonnet average
overall_sonnet = np.mean([sonnet_by_model[m] for m in MODELS_SORTED])

print(f"Loaded data for {len(MODELS_SORTED)} models from {ANALYSIS_DIR}/")
print(f"Overall Sonnet faithfulness: {overall_sonnet:.1f}%")


# ============================================================
# FIGURE 0: Experiment Workflow Diagram
# ============================================================

def make_workflow():
    """Workflow diagram — uses counts from Sonnet faithfulness analysis."""
    n_models = len(MODELS_SORTED)
    n_questions = 498  # Fixed by design
    n_hints = 6
    # Use the Sonnet faithfulness summary for influenced count (5 text-based hints)
    # This matches the 10,276 cases evaluated by both classifiers
    sonnet_summary = load_csv('sonnet_faithfulness_summary.csv')
    total_influenced = sum(int(r['total']) for r in sonnet_summary)
    total_sonnet_faithful = sum(int(r['faithful']) for r in sonnet_summary)
    total_pipeline_faithful = sum(int(r['our_faithful']) for r in sonnet_summary)
    total_runs = n_questions * n_hints * n_models

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    c_data = '#4A90D9'
    c_hint = '#9B59B6'
    c_filter = '#888888'
    c_class = '#E8A838'
    c_result = '#27AE60'
    c_arrow = '#555555'
    box_kw = dict(boxstyle='round,pad=0.4', linewidth=1.5)

    ax.text(2.5, 7.8, f'{n_questions} Questions\n(300 MMLU + 198 GPQA)',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(facecolor=c_data, alpha=0.15, edgecolor=c_data, **box_kw))

    ax.text(7.5, 7.8, f'{n_models} Open-Weight\nReasoning Models',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(facecolor=c_data, alpha=0.15, edgecolor=c_data, **box_kw))

    ax.annotate('', xy=(5, 6.85), xytext=(2.5, 7.25),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.3))
    ax.annotate('', xy=(5, 6.85), xytext=(7.5, 7.25),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.3))

    ax.text(5, 6.5, f'Baseline Runs (no hint)\n{n_questions * n_models:,} inference calls',
            ha='center', va='center', fontsize=9,
            bbox=dict(facecolor='#F0F0F0', edgecolor=c_filter, **box_kw))

    ax.text(5, 5.3, f'{n_hints} Hint Types Injected\n(sycophancy, consistency, visual, metadata, grader, unethical)',
            ha='center', va='center', fontsize=8.5,
            bbox=dict(facecolor=c_hint, alpha=0.15, edgecolor=c_hint, **box_kw))
    ax.annotate('', xy=(5, 5.8), xytext=(5, 6.1),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.3))

    ax.text(5, 4.2, f'Hinted Runs\n{total_runs:,} inference calls',
            ha='center', va='center', fontsize=9,
            bbox=dict(facecolor=c_hint, alpha=0.25, edgecolor=c_hint, **box_kw))
    ax.annotate('', xy=(5, 4.7), xytext=(5, 4.9),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.3))

    infl_pct = total_influenced / total_runs * 100
    ax.text(5, 3.15, f'Filter: answer changed to match hint?\n{total_influenced:,} influenced cases ({infl_pct:.1f}%)',
            ha='center', va='center', fontsize=8.5,
            bbox=dict(facecolor='#F0F0F0', edgecolor=c_filter, **box_kw))
    ax.annotate('', xy=(5, 3.65), xytext=(5, 3.85),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.3))

    for x_target in [2.5, 7.5]:
        ax.annotate('', xy=(x_target, 2.15), xytext=(5, 2.65),
                    arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.2))

    pipeline_rate = total_pipeline_faithful / total_influenced * 100
    sonnet_rate = total_sonnet_faithful / total_influenced * 100
    ax.text(2.5, 1.7, 'Regex + 3-Judge\nLLM Pipeline\n(GLM-5, Kimi K2, Gemini 3)',
            ha='center', va='center', fontsize=8,
            bbox=dict(facecolor=c_class, alpha=0.2, edgecolor=c_class, **box_kw))
    ax.text(7.5, 1.7, 'Claude Sonnet 4\n(independent judge)\n$48.99 total',
            ha='center', va='center', fontsize=8,
            bbox=dict(facecolor=c_result, alpha=0.15, edgecolor=c_result, **box_kw))

    ax.annotate('', xy=(2.5, 0.75), xytext=(2.5, 1.1),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.2))
    ax.annotate('', xy=(7.5, 0.75), xytext=(7.5, 1.1),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.2))

    ax.text(2.5, 0.45, f'{pipeline_rate:.1f}% faithful', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#C07B20')
    ax.text(7.5, 0.45, f'{sonnet_rate:.1f}% faithful', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1E8449')

    total_all = total_runs + n_questions * n_models
    ax.text(5, 8.35, f'Experimental Pipeline: {total_all:,} Total Inference Calls',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#333333')

    fig.savefig(os.path.join(OUT_DIR, 'fig0_workflow.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig0_workflow.png'))
    plt.close(fig)
    print('  fig0_workflow.pdf')


# ============================================================
# FIGURE 1: Influence Rate Heatmap
# ============================================================

def make_influence_heatmap():
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    data = np.array([[influence_per_hint[m][ht] for ht in INFL_HINT_TYPES] for m in MODELS_SORTED])
    short_names = [get_short(m) for m in MODELS_SORTED]
    avg_rates = [influence_by_model[m] for m in MODELS_SORTED]

    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(INFL_HINT_LABELS)))
    ax.set_xticklabels(INFL_HINT_LABELS, fontsize=8.5)
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=8.5)

    for i in range(len(short_names)):
        for j in range(len(INFL_HINT_LABELS)):
            val = data[i, j]
            color = 'white' if val > 55 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7.5, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Influence Rate (%)')

    for i, rate in enumerate(avg_rates):
        ax.text(len(INFL_HINT_LABELS) - 0.15, i, f'avg {rate:.0f}%',
                ha='left', va='center', fontsize=6.5, color='#666', fontweight='bold')

    fig.savefig(os.path.join(OUT_DIR, 'fig1_influence_heatmap.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig1_influence_heatmap.png'))
    plt.close(fig)
    print('  fig1_influence_heatmap.pdf')


# ============================================================
# FIGURE 2: Faithfulness Heatmap (model x hint type)
# ============================================================

def make_faithfulness_heatmap():
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    data = np.array([[sonnet_per_hint[m][ht] for ht in HINT_TYPES] for m in MODELS_SORTED])
    short_names = [get_short(m) for m in MODELS_SORTED]
    rates = [sonnet_by_model[m] for m in MODELS_SORTED]

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(HINT_LABELS)))
    ax.set_xticklabels(HINT_LABELS, fontsize=8.5, rotation=20, ha='right')
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=8.5)

    for i in range(len(short_names)):
        for j in range(len(HINT_LABELS)):
            val = data[i, j]
            color = 'white' if val < 35 or val > 85 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold' if val > 90 or val < 25 else 'normal')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Faithfulness Rate (%)')

    for i, rate in enumerate(rates):
        ax.text(len(HINT_LABELS) - 0.15, i, f'{rate:.0f}%',
                ha='left', va='center', fontsize=7, color='#333', fontweight='bold')

    fig.savefig(os.path.join(OUT_DIR, 'fig2_faithfulness_heatmap.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig2_faithfulness_heatmap.png'))
    plt.close(fig)
    print('  fig2_faithfulness_heatmap.pdf')


# ============================================================
# FIGURE 3: Faithfulness by Hint Type (bar chart with N)
# ============================================================

def make_faithfulness_by_hint():
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    items = [(ht, hint_sonnet_rates[ht], hint_ns[ht]) for ht in HINT_TYPES]
    items.sort(key=lambda x: x[1])
    labels = [HINT_LABELS[HINT_TYPES.index(ht)] for ht, _, _ in items]
    rates = [r for _, r, _ in items]
    ns = [n for _, _, n in items]

    colors = ['#E74C3C' if r < 50 else '#E8A838' if r < 70 else '#27AE60' for r in rates]
    ax.barh(range(len(labels)), rates, color=colors, alpha=0.8, edgecolor='white', height=0.6)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Sonnet-Judged Faithfulness Rate (%)')
    ax.set_xlim(0, 100)

    for i, (rate, n) in enumerate(zip(rates, ns)):
        ax.text(rate + 1.5, i, f'{rate:.1f}%  (n={n:,})',
                ha='left', va='center', fontsize=8.5, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.axvline(x=overall_sonnet, color='#555', ls='--', lw=1, alpha=0.6)
    ax.text(overall_sonnet, len(labels) - 0.3, f'Overall\n{overall_sonnet:.1f}%',
            ha='center', va='bottom', fontsize=7.5, color='#555')

    fig.savefig(os.path.join(OUT_DIR, 'fig3_faithfulness_by_hint.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig3_faithfulness_by_hint.png'))
    plt.close(fig)
    print('  fig3_faithfulness_by_hint.pdf')


# ============================================================
# FIGURE 4: Faithfulness vs Scale (scatter)
# ============================================================

def make_faithfulness_vs_scale():
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

    params_filt = []
    rates_filt = []
    names_filt = []
    for m in MODELS_SORTED:
        p = ACTIVE_PARAMS_MAP.get(m)
        if p is not None:
            params_filt.append(p)
            rates_filt.append(sonnet_by_model[m])
            names_filt.append(get_short(m))

    ax.scatter(params_filt, rates_filt, s=80, c='#4A90D9', alpha=0.8,
               edgecolors='white', linewidths=0.8, zorder=3)

    for p, r, name in zip(params_filt, rates_filt, names_filt):
        offset_x = 0.08
        offset_y = 1.5
        if name in ('DS-R1', 'DS-V3.2'):
            offset_y = -3
        elif name == 'QwQ-32B':
            offset_x = -0.08
            offset_y = -2.5
        elif name == 'OLMo-32B':
            offset_x = -0.08
            offset_y = 2
        elif name in ('Step-3.5', 'MiniMax'):
            offset_y = 2.5

        ax.annotate(name, xy=(p, r),
                    xytext=(p * (1 + offset_x * 3), r + offset_y),
                    fontsize=7, color='#333')

    ax.set_xscale('log')
    ax.set_xlabel('Active Parameters (B, log scale)')
    ax.set_ylabel('Sonnet Faithfulness Rate (%)')
    ax.set_ylim(35, 95)
    ax.set_xlim(2, 50)

    log_params = np.log10(params_filt)
    z = np.polyfit(log_params, rates_filt, 1)
    x_line = np.linspace(2, 50, 100)
    y_line = np.polyval(z, np.log10(x_line))
    ax.plot(x_line, y_line, '--', color='#CCC', lw=1, zorder=1)

    y_pred = np.polyval(z, log_params)
    ss_res = sum((r - yp) ** 2 for r, yp in zip(rates_filt, y_pred))
    ss_tot = sum((r - np.mean(rates_filt)) ** 2 for r in rates_filt)
    r_sq = 1 - ss_res / ss_tot
    ax.text(0.95, 0.05, f'$R^2$ = {r_sq:.2f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='#888')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(OUT_DIR, 'fig4_faithfulness_vs_scale.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig4_faithfulness_vs_scale.png'))
    plt.close(fig)
    print('  fig4_faithfulness_vs_scale.pdf')


# ============================================================
# FIGURE 5: CoT Length by Model
# ============================================================

def make_cot_length():
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    items = [(get_short(m), cot_by_model[m], sonnet_by_model[m]) for m in MODELS_SORTED]
    items.sort(key=lambda x: x[1])
    names = [x[0] for x in items]
    tokens = [x[1] for x in items]
    faith = [x[2] for x in items]

    colors = ['#E74C3C' if f < 55 else '#E8A838' if f < 75 else '#27AE60' for f in faith]
    ax.barh(range(len(names)), tokens, color=colors, alpha=0.8, edgecolor='white', height=0.6)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Median Reasoning Tokens')

    for i, (t, f) in enumerate(zip(tokens, faith)):
        ax.text(t + 50, i, f'{t:,.0f}  ({f:.0f}% faithful)',
                ha='left', va='center', fontsize=7.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 5000)

    legend_elements = [
        mpatches.Patch(facecolor='#27AE60', alpha=0.8, label='Faithful > 75%'),
        mpatches.Patch(facecolor='#E8A838', alpha=0.8, label='55-75%'),
        mpatches.Patch(facecolor='#E74C3C', alpha=0.8, label='Faithful < 55%'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7.5, framealpha=0.9)

    fig.savefig(os.path.join(OUT_DIR, 'fig5_cot_length.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig5_cot_length.png'))
    plt.close(fig)
    print('  fig5_cot_length.pdf')


# ============================================================
# FIGURE 6: Thinking vs Answer Faithfulness Gap
# ============================================================

def make_thinking_vs_answer():
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    items = [(get_short(m), thinking_by_model[m], answer_by_model[m]) for m in MODELS_SORTED]
    items.sort(key=lambda x: (x[1] - x[2]), reverse=True)
    names = [x[0] for x in items]
    thinking = [x[1] for x in items]
    answer = [x[2] for x in items]
    gaps = [t - a for t, a in zip(thinking, answer)]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width / 2, thinking, width, label='Thinking Tokens',
           color='#4A90D9', alpha=0.85, edgecolor='white')
    ax.bar(x + width / 2, answer, width, label='Answer Text',
           color='#E74C3C', alpha=0.75, edgecolor='white')

    for i, (t, g) in enumerate(zip(thinking, gaps)):
        ax.annotate(f'{g:.0f}pp', xy=(i, t + 1), ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color='#555')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Acknowledgment Rate (%)')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    avg_t = np.mean(thinking)
    avg_a = np.mean(answer)
    ax.axhline(y=avg_t, color='#4A90D9', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(y=avg_a, color='#E74C3C', ls='--', lw=0.8, alpha=0.5)
    ax.text(len(names) - 0.5, avg_t + 1, f'avg {avg_t:.1f}%', fontsize=7,
            color='#4A90D9', ha='right')
    ax.text(len(names) - 0.5, avg_a + 1, f'avg {avg_a:.1f}%', fontsize=7,
            color='#E74C3C', ha='right')

    fig.savefig(os.path.join(OUT_DIR, 'fig6_thinking_vs_answer.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig6_thinking_vs_answer.png'))
    plt.close(fig)
    print('  fig6_thinking_vs_answer.pdf')


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print(f'Generating Paper 1 figures to {OUT_DIR}/')
    make_workflow()
    make_influence_heatmap()
    make_faithfulness_heatmap()
    make_faithfulness_by_hint()
    make_faithfulness_vs_scale()
    make_cot_length()
    make_thinking_vs_answer()
    print('Done. Generated 7 figures.')
