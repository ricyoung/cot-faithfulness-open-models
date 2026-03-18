"""Generate all figures for Paper 2: Classifier Sensitivity."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper2', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Use a clean style
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
# DATA (from results CSVs)
# ============================================================

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

# Short names for plots
SHORT_NAMES = [
    'DS-V3.2',
    'GPT-OSS',
    'OLMo-32B',
    'Step-3.5',
    'DS-R1',
    'MiniMax',
    'Qwen3.5',
    'ERNIE-4.5',
    'Nemotron',
    'OLMo-7B',
    'QwQ-32B',
    'Seed-1.6',
]

N_CASES = [899, 769, 997, 750, 1193, 554, 1308, 900, 732, 580, 982, 612]

REGEX_RATES = [94.4, 78.4, 66.5, 93.2, 91.6, 85.5, 97.5, 67.6, 37.5, 70.9, 55.1, 26.5]
PIPELINE_RATES = [97.6, 94.7, 71.9, 96.0, 94.8, 91.2, 98.9, 75.1, 67.4, 80.2, 66.5, 37.1]
SONNET_RATES = [89.9, 84.9, 81.0, 75.3, 74.8, 73.1, 68.3, 62.8, 60.9, 56.9, 56.3, 39.7]

HINT_TYPES = ['Sycophancy', 'Consistency', 'Unethical', 'Metadata', 'Grader']
HINT_PIPELINE = [97.3, 68.6, 88.4, 60.4, 80.6]
HINT_SONNET = [53.9, 35.5, 79.4, 69.9, 77.7]
HINT_GAP = [p - s for p, s in zip(HINT_PIPELINE, HINT_SONNET)]

# Ranks (sorted by each classifier's rate, rank 1 = highest)
RANK_PIPELINE = [2, 6, 9, 3, 4, 5, 1, 8, 10, 7, 11, 12]
RANK_SONNET = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# ============================================================
# FIGURE 1: Workflow Diagram
# ============================================================

def make_workflow():
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Colors
    c_data = '#4A90D9'
    c_regex = '#E8A838'
    c_pipeline = '#E8A838'
    c_sonnet = '#6BBF6B'
    c_result = '#D95B5B'
    c_arrow = '#555555'

    box_kw = dict(boxstyle='round,pad=0.4', linewidth=1.5)

    # Top: Data source
    ax.text(5, 6.3, '41,832 Hinted Inference Runs\n(12 models × 498 questions × ~7 hint types)',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(facecolor=c_data, alpha=0.15, edgecolor=c_data, **box_kw))

    # Arrow down to filter
    ax.annotate('', xy=(5, 5.35), xytext=(5, 5.85),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.5))

    # Filter box
    ax.text(5, 5.05, 'Filter: Answer changed?\n10,276 influenced cases',
            ha='center', va='center', fontsize=9,
            bbox=dict(facecolor='#F0F0F0', edgecolor='#888888', **box_kw))

    # Arrow splits into three
    for x_target in [1.8, 5.0, 8.2]:
        ax.annotate('', xy=(x_target, 3.85), xytext=(5, 4.6),
                    arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.2))

    # Three classifiers
    ax.text(1.8, 3.4, 'Classifier 1\nRegex-only\n(38 patterns)',
            ha='center', va='center', fontsize=8.5,
            bbox=dict(facecolor=c_regex, alpha=0.15, edgecolor=c_regex, **box_kw))

    ax.text(5.0, 3.4, 'Classifier 2\nRegex + Ollama\n(3-judge majority)',
            ha='center', va='center', fontsize=8.5,
            bbox=dict(facecolor=c_pipeline, alpha=0.3, edgecolor=c_pipeline, **box_kw))

    ax.text(8.2, 3.4, 'Classifier 3\nClaude Sonnet 4\n(independent judge)',
            ha='center', va='center', fontsize=8.5,
            bbox=dict(facecolor=c_sonnet, alpha=0.2, edgecolor=c_sonnet, **box_kw))

    # Arrows down to results
    for x_target in [1.8, 5.0, 8.2]:
        ax.annotate('', xy=(x_target, 2.15), xytext=(x_target, 2.85),
                    arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.2))

    # Results
    ax.text(1.8, 1.75, '74.4%', ha='center', va='center', fontsize=14,
            fontweight='bold', color=c_regex)
    ax.text(5.0, 1.75, '82.6%', ha='center', va='center', fontsize=14,
            fontweight='bold', color='#C07B20')
    ax.text(8.2, 1.75, '69.7%', ha='center', va='center', fontsize=14,
            fontweight='bold', color='#3A8A3A')

    # Bottom label
    ax.text(5, 0.8, 'Same 10,276 cases → three different faithfulness rates',
            ha='center', va='center', fontsize=10, fontstyle='italic',
            bbox=dict(facecolor=c_result, alpha=0.1, edgecolor=c_result, **box_kw))

    # Cost labels
    ax.text(1.8, 1.25, '$0 (instant)', ha='center', va='center', fontsize=7.5, color='#666')
    ax.text(5.0, 1.25, '$0 (local GPU)', ha='center', va='center', fontsize=7.5, color='#666')
    ax.text(8.2, 1.25, '$48.99 (API)', ha='center', va='center', fontsize=7.5, color='#666')

    fig.savefig(os.path.join(OUT_DIR, 'workflow.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'workflow.png'))
    plt.close(fig)
    print('  workflow.pdf')


# ============================================================
# FIGURE 2: Classifier Gap by Hint Type
# ============================================================

def make_hint_gap():
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    # Sort by gap magnitude (descending)
    order = sorted(range(len(HINT_GAP)), key=lambda i: abs(HINT_GAP[i]), reverse=True)
    labels = [HINT_TYPES[i] for i in order]
    gaps = [HINT_GAP[i] for i in order]
    pipeline_vals = [HINT_PIPELINE[i] for i in order]
    sonnet_vals = [HINT_SONNET[i] for i in order]

    x = np.arange(len(labels))
    width = 0.3

    bars_p = ax.bar(x - width/2, pipeline_vals, width, label='Pipeline',
                     color='#E8A838', alpha=0.8, edgecolor='white')
    bars_s = ax.bar(x + width/2, sonnet_vals, width, label='Sonnet',
                     color='#6BBF6B', alpha=0.8, edgecolor='white')

    # Add gap annotations
    for i, (p, s, g) in enumerate(zip(pipeline_vals, sonnet_vals, gaps)):
        mid = (p + s) / 2
        sign = '+' if g > 0 else ''
        ax.annotate(f'{sign}{g:.1f} pp',
                    xy=(i, max(p, s) + 1.5), ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='#D95B5B')

    ax.set_ylabel('Faithfulness Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(OUT_DIR, 'hint_gap.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'hint_gap.png'))
    plt.close(fig)
    print('  hint_gap.pdf')


# ============================================================
# FIGURE 3: Model Ranking Comparison (Slope/Bump Chart)
# ============================================================

def make_rank_comparison():
    fig, ax = plt.subplots(1, 1, figsize=(5, 5.5))

    # Sort models by pipeline rank for consistent ordering
    n = len(SHORT_NAMES)

    x_left = 0.2
    x_right = 0.8

    # Color by magnitude of rank change
    for i in range(n):
        rank_p = RANK_PIPELINE[i]
        rank_s = RANK_SONNET[i]
        delta = abs(rank_p - rank_s)

        if delta >= 5:
            color = '#D95B5B'
            lw = 2.5
            alpha = 1.0
        elif delta >= 3:
            color = '#E8A838'
            lw = 1.8
            alpha = 0.9
        else:
            color = '#888888'
            lw = 1.2
            alpha = 0.6

        ax.plot([x_left, x_right], [rank_p, rank_s],
                color=color, lw=lw, alpha=alpha, zorder=2)

        # Left labels (pipeline rank)
        ax.text(x_left - 0.03, rank_p, f'{SHORT_NAMES[i]} ({PIPELINE_RATES[i]:.0f}%)',
                ha='right', va='center', fontsize=7.5)

        # Right labels (sonnet rank)
        ax.text(x_right + 0.03, rank_s, f'{SHORT_NAMES[i]} ({SONNET_RATES[i]:.0f}%)',
                ha='left', va='center', fontsize=7.5)

        # Dots
        ax.scatter([x_left], [rank_p], color=color, s=30, zorder=3, edgecolors='white', linewidths=0.5)
        ax.scatter([x_right], [rank_s], color=color, s=30, zorder=3, edgecolors='white', linewidths=0.5)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(['Pipeline Rank', 'Sonnet Rank'], fontsize=10, fontweight='bold')
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([str(i) for i in range(1, n + 1)], fontsize=8)
    ax.set_ylabel('Rank (1 = most faithful)')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Legend
    legend_elements = [
        mlines.Line2D([], [], color='#D95B5B', lw=2.5, label='Large shift (5+ ranks)'),
        mlines.Line2D([], [], color='#E8A838', lw=1.8, label='Moderate shift (3-4 ranks)'),
        mlines.Line2D([], [], color='#888888', lw=1.2, label='Small shift (0-2 ranks)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=8,
              framealpha=0.9, ncol=1)

    # Add Spearman annotation
    ax.text(0.5, n + 0.3, r'Spearman $\rho$ = 0.64 ($p$ = 0.024)',
            ha='center', va='center', fontsize=9, fontstyle='italic',
            transform=ax.transData)

    fig.savefig(os.path.join(OUT_DIR, 'rank_comparison.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'rank_comparison.png'))
    plt.close(fig)
    print('  rank_comparison.pdf')


# ============================================================
# FIGURE 4: Scatter — Pipeline vs Sonnet Rate
# ============================================================

def make_scatter():
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    # Diagonal line (perfect agreement)
    ax.plot([20, 100], [20, 100], '--', color='#CCCCCC', lw=1, zorder=1, label='Perfect agreement')

    # Size by N
    sizes = [n / 15 for n in N_CASES]

    scatter = ax.scatter(SONNET_RATES, PIPELINE_RATES, s=sizes, c='#4A90D9',
                         alpha=0.7, edgecolors='white', linewidths=0.8, zorder=3)

    # Label each point
    for i in range(len(SHORT_NAMES)):
        offset_x = 1.5
        offset_y = -1.5
        # Adjust overlapping labels
        if SHORT_NAMES[i] == 'DS-R1':
            offset_y = 2
        elif SHORT_NAMES[i] == 'Step-3.5':
            offset_y = -3
        elif SHORT_NAMES[i] == 'MiniMax':
            offset_y = 2.5
        elif SHORT_NAMES[i] == 'OLMo-32B':
            offset_x = -3
            offset_y = 1
        elif SHORT_NAMES[i] == 'Seed-1.6':
            offset_x = 2
            offset_y = 2

        ax.annotate(SHORT_NAMES[i],
                    xy=(SONNET_RATES[i], PIPELINE_RATES[i]),
                    xytext=(SONNET_RATES[i] + offset_x, PIPELINE_RATES[i] + offset_y),
                    fontsize=7, color='#333333',
                    arrowprops=dict(arrowstyle='-', color='#CCCCCC', lw=0.5) if abs(offset_x) > 2 or abs(offset_y) > 2.5 else None)

    ax.set_xlabel('Sonnet Faithfulness Rate (%)')
    ax.set_ylabel('Pipeline Faithfulness Rate (%)')
    ax.set_xlim(30, 100)
    ax.set_ylim(30, 105)

    # Shade region where pipeline > sonnet (above diagonal)
    ax.fill_between([30, 100], [30, 100], [105, 105], alpha=0.04, color='#E8A838')
    ax.text(50, 98, 'Pipeline more generous', fontsize=7.5, color='#C07B20', fontstyle='italic')

    # Shade region where sonnet > pipeline (below diagonal)
    ax.fill_between([30, 100], [30, 100], [30, 30], alpha=0.04, color='#6BBF6B')
    ax.text(75, 40, 'Sonnet more generous', fontsize=7.5, color='#3A8A3A', fontstyle='italic')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(OUT_DIR, 'scatter_agreement.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'scatter_agreement.png'))
    plt.close(fig)
    print('  scatter_agreement.pdf')


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print(f'Generating Paper 2 figures to {OUT_DIR}/')
    make_workflow()
    make_hint_gap()
    make_rank_comparison()
    make_scatter()
    print('Done.')
