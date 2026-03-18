"""Generate all figures for Paper 1: Lie to Me — CoT Faithfulness Study."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
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
# DATA (from results/analysis CSVs)
# ============================================================

# Models sorted by Sonnet faithfulness (descending)
MODELS = [
    'DeepSeek-V3.2-Speciale',
    'GPT-OSS-120B',
    'OLMo-3.1-32B',
    'Step-3.5-Flash',
    'DeepSeek-R1',
    'MiniMax-M2.5',
    'Qwen3.5-27B',
    'ERNIE-4.5-21B',
    'Nemotron-Nano-9B',
    'OLMo-3-7B',
    'QwQ-32B',
    'Seed-1.6-Flash',
]

SHORT_NAMES = [
    'DS-V3.2', 'GPT-OSS', 'OLMo-32B', 'Step-3.5',
    'DS-R1', 'MiniMax', 'Qwen3.5', 'ERNIE-4.5',
    'Nemotron', 'OLMo-7B', 'QwQ-32B', 'Seed-1.6',
]

# Active parameters (billions) for scaling analysis
ACTIVE_PARAMS = [37, 5.1, 32, 11, 37, 10, 27, 3, 9, 7, 32, None]  # Seed undisclosed

# Sonnet faithfulness rates (primary)
SONNET_RATES = [89.9, 84.9, 81.0, 75.3, 74.8, 73.1, 68.3, 62.8, 60.9, 56.9, 56.3, 39.7]

# Pipeline faithfulness rates (secondary)
PIPELINE_RATES = [97.6, 94.7, 71.9, 96.0, 94.8, 91.2, 98.9, 75.1, 67.4, 80.2, 66.5, 37.1]

# Baseline accuracy (overall %)
ACCURACY = [90.9, 85.0, 73.3, 87.0, 86.9, 86.8, 84.6, 72.5, 72.4, 67.7, 79.0, 71.8]

# Average influence rate (%)
INFLUENCE_RATE = [33.2, 28.6, 34.0, 27.7, 41.2, 20.2, 44.6, 31.0, 24.9, 26.3, 36.8, 22.4]

# Median reasoning tokens
MEDIAN_REASONING = [1393, 951, 3290, 1627, 1393, 827, 2888, 2305, 1209, 3818, 1390, 993]

# Thinking vs answer faithfulness (%)
THINKING_RATE = [97.8, 93.0, 89.6, 97.8, 95.5, 93.8, 99.3, 82.8, 67.1, 86.4, 75.9, 59.5]
ANSWER_RATE = [5.3, 0.0, 1.3, 3.0, 72.5, 9.2, 79.9, 46.3, 35.3, 27.4, 33.4, 29.8]

# Hint-type data (Sonnet rates %)
HINT_TYPES = ['Consistency', 'Sycophancy', 'Metadata', 'Grader', 'Unethical']
HINT_SONNET = [35.5, 53.9, 69.9, 77.7, 79.4]
HINT_N = [650, 2034, 1531, 2829, 3232]

# Per-model per-hint Sonnet faithfulness (rows=models in MODELS order, cols=hint types in HINT_TYPES order)
FAITH_HEATMAP = [
    # Consist  Syco    Meta    Grader  Uneth
    [37.0,  67.1, 76.8, 94.0, 98.1],   # DS-V3.2
    [16.7,  45.6, 85.0, 96.0, 82.3],   # GPT-OSS
    [42.0,  78.4, 75.7, 74.0, 92.9],   # OLMo-32B
    [13.9,  48.0, 79.8, 66.5, 96.9],   # Step-3.5
    [31.0,  62.1, 70.9, 81.3, 87.8],   # DS-R1
    [34.8,  68.2, 77.1, 89.6, 51.5],   # MiniMax
    [55.6,  24.2, 88.6, 96.4, 82.7],   # Qwen3.5
    [29.9,  62.5, 76.2, 68.8, 58.8],   # ERNIE-4.5
    [58.1,  54.4, 74.8, 59.0, 58.6],   # Nemotron
    [25.0,  57.5, 59.3, 62.5, 63.7],   # OLMo-7B
    [35.2,  64.9, 56.5, 53.7, 59.2],   # QwQ-32B
    [32.5,  53.2, 26.7, 43.8, 42.5],   # Seed-1.6
]

# Influence rate per model per hint (for heatmap)
INFL_HEATMAP = [
    # Syco    Consist  VisPat  Meta    Grader  Uneth
    [17.2, 4.9, 0.5, 21.2, 66.2, 89.4],  # DS-V3.2
    [13.8, 3.0, 1.3, 21.7, 85.5, 46.4],  # GPT-OSS
    [30.8, 13.9, 4.3, 28.8, 38.6, 88.0],  # OLMo-32B
    [27.0, 7.0, 1.2, 19.0, 39.3, 72.9],  # Step-3.5
    [63.1, 10.9, 2.1, 30.4, 58.9, 81.6],  # DS-R1
    [16.8, 4.1, 2.2, 14.7, 51.8, 31.9],  # MiniMax
    [79.6, 9.1, 3.3, 34.2, 66.9, 74.3],  # Qwen3.5
    [46.5, 15.6, 4.9, 37.3, 35.0, 47.0],  # ERNIE-4.5
    [28.1, 9.7, 2.9, 28.0, 36.7, 43.9],  # Nemotron
    [31.3, 12.6, 5.2, 20.6, 44.3, 44.1],  # OLMo-7B
    [39.6, 17.2, 3.5, 39.7, 53.1, 67.8],  # QwQ-32B
    [23.5, 17.6, 4.8, 31.3, 40.9, 16.4],  # Seed-1.6
]
INFL_HINT_LABELS = ['Sycophancy', 'Consistency', 'Visual\nPattern', 'Metadata', 'Grader', 'Unethical']


# ============================================================
# FIGURE 0: Experiment Workflow Diagram
# ============================================================

def make_workflow():
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

    # Row 1: Data sources
    ax.text(2.5, 7.8, '498 Questions\n(300 MMLU + 198 GPQA)',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(facecolor=c_data, alpha=0.15, edgecolor=c_data, **box_kw))

    ax.text(7.5, 7.8, '12 Open-Weight\nReasoning Models',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(facecolor=c_data, alpha=0.15, edgecolor=c_data, **box_kw))

    # Arrows to baseline
    ax.annotate('', xy=(5, 6.85), xytext=(2.5, 7.25),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.3))
    ax.annotate('', xy=(5, 6.85), xytext=(7.5, 7.25),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.3))

    # Row 2: Baseline runs
    ax.text(5, 6.5, 'Baseline Runs (no hint)\n5,976 inference calls',
            ha='center', va='center', fontsize=9,
            bbox=dict(facecolor='#F0F0F0', edgecolor=c_filter, **box_kw))

    # Hint injection box
    ax.text(5, 5.3, '6 Hint Types Injected\n(sycophancy, consistency, visual, metadata, grader, unethical)',
            ha='center', va='center', fontsize=8.5,
            bbox=dict(facecolor=c_hint, alpha=0.15, edgecolor=c_hint, **box_kw))
    ax.annotate('', xy=(5, 5.8), xytext=(5, 6.1),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.3))

    # Hinted runs
    ax.text(5, 4.2, 'Hinted Runs\n35,856 inference calls',
            ha='center', va='center', fontsize=9,
            bbox=dict(facecolor=c_hint, alpha=0.25, edgecolor=c_hint, **box_kw))
    ax.annotate('', xy=(5, 4.7), xytext=(5, 4.9),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.3))

    # Filter
    ax.text(5, 3.15, 'Filter: answer changed to match hint?\n10,276 influenced cases (28.7%)',
            ha='center', va='center', fontsize=8.5,
            bbox=dict(facecolor='#F0F0F0', edgecolor=c_filter, **box_kw))
    ax.annotate('', xy=(5, 3.65), xytext=(5, 3.85),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.3))

    # Split to classifiers
    for x_target in [2.5, 7.5]:
        ax.annotate('', xy=(x_target, 2.15), xytext=(5, 2.65),
                    arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.2))

    # Two classifiers
    ax.text(2.5, 1.7, 'Regex + 3-Judge\nLLM Pipeline\n(GLM-5, Kimi K2, Gemini 3)',
            ha='center', va='center', fontsize=8,
            bbox=dict(facecolor=c_class, alpha=0.2, edgecolor=c_class, **box_kw))

    ax.text(7.5, 1.7, 'Claude Sonnet 4\n(independent judge)\n$48.99 total',
            ha='center', va='center', fontsize=8,
            bbox=dict(facecolor=c_result, alpha=0.15, edgecolor=c_result, **box_kw))

    # Results
    ax.annotate('', xy=(2.5, 0.75), xytext=(2.5, 1.1),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.2))
    ax.annotate('', xy=(7.5, 0.75), xytext=(7.5, 1.1),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.2))

    ax.text(2.5, 0.45, '82.6% faithful', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#C07B20')
    ax.text(7.5, 0.45, '69.7% faithful', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1E8449')

    # Total at top
    ax.text(5, 8.35, 'Experimental Pipeline: 41,832 Total Inference Calls',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#333333')

    fig.savefig(os.path.join(OUT_DIR, 'fig0_workflow.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig0_workflow.png'))
    plt.close(fig)
    print('  fig0_workflow.pdf')


# ============================================================
# FIGURE 1: Influence Rate Heatmap
# ============================================================

def make_influence_heatmap():
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    data = np.array(INFL_HEATMAP)
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(INFL_HINT_LABELS)))
    ax.set_xticklabels(INFL_HINT_LABELS, fontsize=8.5)
    ax.set_yticks(range(len(SHORT_NAMES)))
    ax.set_yticklabels(SHORT_NAMES, fontsize=8.5)

    # Add text annotations
    for i in range(len(SHORT_NAMES)):
        for j in range(len(INFL_HINT_LABELS)):
            val = data[i, j]
            color = 'white' if val > 55 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=7.5, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Influence Rate (%)')

    # Add average column annotation
    for i, rate in enumerate(INFLUENCE_RATE):
        ax.text(len(INFL_HINT_LABELS) - 0.15, i, f'avg {rate:.0f}%',
                ha='left', va='center', fontsize=6.5, color='#666',
                fontweight='bold')

    fig.savefig(os.path.join(OUT_DIR, 'fig1_influence_heatmap.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig1_influence_heatmap.png'))
    plt.close(fig)
    print('  fig1_influence_heatmap.pdf')


# ============================================================
# FIGURE 2: Faithfulness Heatmap (model x hint type)
# ============================================================

def make_faithfulness_heatmap():
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    data = np.array(FAITH_HEATMAP)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(HINT_TYPES)))
    ax.set_xticklabels(HINT_TYPES, fontsize=8.5, rotation=20, ha='right')
    ax.set_yticks(range(len(SHORT_NAMES)))
    ax.set_yticklabels(SHORT_NAMES, fontsize=8.5)

    # Add text annotations
    for i in range(len(SHORT_NAMES)):
        for j in range(len(HINT_TYPES)):
            val = data[i, j]
            color = 'white' if val < 35 or val > 85 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold' if val > 90 or val < 25 else 'normal')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Faithfulness Rate (%)')

    # Add overall column annotation
    for i, rate in enumerate(SONNET_RATES):
        ax.text(len(HINT_TYPES) - 0.15, i, f'{rate:.0f}%',
                ha='left', va='center', fontsize=7, color='#333',
                fontweight='bold')

    fig.savefig(os.path.join(OUT_DIR, 'fig2_faithfulness_heatmap.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig2_faithfulness_heatmap.png'))
    plt.close(fig)
    print('  fig2_faithfulness_heatmap.pdf')


# ============================================================
# FIGURE 3: Faithfulness by Hint Type (bar chart with N)
# ============================================================

def make_faithfulness_by_hint():
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    # Sort by Sonnet rate (ascending for visual impact)
    order = sorted(range(len(HINT_SONNET)), key=lambda i: HINT_SONNET[i])
    labels = [HINT_TYPES[i] for i in order]
    rates = [HINT_SONNET[i] for i in order]
    ns = [HINT_N[i] for i in order]

    colors = ['#E74C3C' if r < 50 else '#E8A838' if r < 70 else '#27AE60' for r in rates]

    bars = ax.barh(range(len(labels)), rates, color=colors, alpha=0.8, edgecolor='white', height=0.6)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Sonnet-Judged Faithfulness Rate (%)')
    ax.set_xlim(0, 100)

    # Add rate and N labels
    for i, (rate, n) in enumerate(zip(rates, ns)):
        ax.text(rate + 1.5, i, f'{rate:.1f}%  (n={n:,})',
                ha='left', va='center', fontsize=8.5, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add vertical line at overall average
    ax.axvline(x=69.7, color='#555', ls='--', lw=1, alpha=0.6)
    ax.text(69.7, len(labels) - 0.3, 'Overall\n69.7%', ha='center', va='bottom',
            fontsize=7.5, color='#555')

    fig.savefig(os.path.join(OUT_DIR, 'fig3_faithfulness_by_hint.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig3_faithfulness_by_hint.png'))
    plt.close(fig)
    print('  fig3_faithfulness_by_hint.pdf')


# ============================================================
# FIGURE 4: Faithfulness vs Scale (scatter)
# ============================================================

def make_faithfulness_vs_scale():
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

    # Filter out Seed (undisclosed params)
    params_filt = []
    rates_filt = []
    names_filt = []
    for i in range(len(MODELS)):
        if ACTIVE_PARAMS[i] is not None:
            params_filt.append(ACTIVE_PARAMS[i])
            rates_filt.append(SONNET_RATES[i])
            names_filt.append(SHORT_NAMES[i])

    ax.scatter(params_filt, rates_filt, s=80, c='#4A90D9', alpha=0.8,
               edgecolors='white', linewidths=0.8, zorder=3)

    # Label each point
    for i, (p, r, name) in enumerate(zip(params_filt, rates_filt, names_filt)):
        offset_x = 0.08  # log scale offset
        offset_y = 1.5
        if name == 'DS-R1':
            offset_y = -3
        elif name == 'DS-V3.2':
            offset_y = -3
        elif name == 'QwQ-32B':
            offset_x = -0.08
            offset_y = -2.5
        elif name == 'OLMo-32B':
            offset_x = -0.08
            offset_y = 2
        elif name == 'Step-3.5':
            offset_y = 2.5
        elif name == 'MiniMax':
            offset_y = 2.5

        ax.annotate(name, xy=(p, r),
                    xytext=(p * (1 + offset_x * 3), r + offset_y),
                    fontsize=7, color='#333')

    ax.set_xscale('log')
    ax.set_xlabel('Active Parameters (B, log scale)')
    ax.set_ylabel('Sonnet Faithfulness Rate (%)')
    ax.set_ylim(35, 95)
    ax.set_xlim(2, 50)

    # Add trend line
    log_params = np.log10(params_filt)
    z = np.polyfit(log_params, rates_filt, 1)
    x_line = np.linspace(2, 50, 100)
    y_line = np.polyval(z, np.log10(x_line))
    ax.plot(x_line, y_line, '--', color='#CCC', lw=1, zorder=1)

    # Compute R^2
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

    # Sort by median reasoning tokens
    order = sorted(range(len(MEDIAN_REASONING)), key=lambda i: MEDIAN_REASONING[i])
    names = [SHORT_NAMES[i] for i in order]
    tokens = [MEDIAN_REASONING[i] for i in order]
    faith = [SONNET_RATES[i] for i in order]

    # Color by faithfulness
    colors = ['#E74C3C' if f < 55 else '#E8A838' if f < 75 else '#27AE60' for f in faith]

    bars = ax.barh(range(len(names)), tokens, color=colors, alpha=0.8,
                    edgecolor='white', height=0.6)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Median Reasoning Tokens')

    # Add token count labels
    for i, (t, f) in enumerate(zip(tokens, faith)):
        ax.text(t + 50, i, f'{t:,}  ({f:.0f}% faithful)',
                ha='left', va='center', fontsize=7.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 5000)

    # Legend
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

    # Sort by gap (descending)
    gaps = [t - a for t, a in zip(THINKING_RATE, ANSWER_RATE)]
    order = sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True)

    names = [SHORT_NAMES[i] for i in order]
    thinking = [THINKING_RATE[i] for i in order]
    answer = [ANSWER_RATE[i] for i in order]
    gap_sorted = [gaps[i] for i in order]

    x = np.arange(len(names))
    width = 0.35

    bars_t = ax.bar(x - width / 2, thinking, width, label='Thinking Tokens',
                     color='#4A90D9', alpha=0.85, edgecolor='white')
    bars_a = ax.bar(x + width / 2, answer, width, label='Answer Text',
                     color='#E74C3C', alpha=0.75, edgecolor='white')

    # Gap annotations
    for i, (t, a, g) in enumerate(zip(thinking, answer, gap_sorted)):
        ax.annotate(f'{g:.0f}pp',
                    xy=(i, t + 1), ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color='#555')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Faithfulness Rate (%)')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add averages
    avg_t = np.mean(THINKING_RATE)
    avg_a = np.mean(ANSWER_RATE)
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
