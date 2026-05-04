#!/usr/bin/env python3
"""
plot_results.py — NeuSym-HLS result plots.
Usage: python plot_results.py
Output: outputs/plots/
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

OUT_DIR = Path(__file__).parent / 'outputs' / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def mpl(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

MC = {k: mpl(v) for k, v in {
    'blue':   '#1D4ED8',
    'lblue':  '#93C5FD',
    'orange': '#EA580C',
    'lorange':'#FED7AA',
    'teal':   '#0F766E',
    'lteal':  '#99F6E4',
    'red':    '#B91C1C',
    'gray':   '#6B7280',
    'lgray':  '#F3F4F6',
    'black':  '#111827',
}.items()}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.2,
    'grid.linestyle':    '--',
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
})

# ── Data ───────────────────────────────────────────────────────────────────────
LABELS = [
    'Baseline\nfc3',
    'Baseline\nfc2+fc3',
    'SR-1L\nPOL',
    'SR-1L\nSRL',
    'SR-1L\nSCE',
    'SR-2L\nPOL',
    'SR-2L\nSRL',
    'SR-2L\nSCE',
]
LUT = [943,  2035, 1363, 1429, 1643,  640, 1448, 4266]
FF  = [498,  4813,  683,  707,  853,  278,  690, 2493]
DSP = [5,    5,     7,    7,    10,    2,    7,   20  ]

BASELINE_ACC = 0.961   # fill from outputs/results/baseline_metrics.json
ACC = [
    BASELINE_ACC, BASELINE_ACC,
    0.950829, 0.949003, 0.950688,
    0.925260, 0.933830, 0.927508,
]

# bar color per model
BAR_COLORS = [
    MC['orange'],   # Baseline fc3
    MC['lorange'],  # Baseline fc2+fc3
    MC['blue'],     # SR-1L-POL
    MC['blue'],     # SR-1L-SRL
    MC['blue'],     # SR-1L-SCE
    MC['teal'],     # SR-2L-POL
    MC['teal'],     # SR-2L-SRL
    MC['teal'],     # SR-2L-SCE
]

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved: {path}')


# ── Plot 1: Total Resources (LUT + FF stacked) vs Accuracy ────────────────────
def plot_area_vs_accuracy():
    fig, ax1 = plt.subplots(figsize=(13, 6))
    x = np.arange(len(LABELS))
    w = 0.55

    # Stacked bars: LUT (bottom) + FF (top)
    b1 = ax1.bar(x, LUT, width=w, color=BAR_COLORS, alpha=0.9,
                 edgecolor='white', linewidth=0.8, zorder=3, label='LUT')
    b2 = ax1.bar(x, FF,  width=w, bottom=LUT, color=BAR_COLORS, alpha=0.45,
                 edgecolor='white', linewidth=0.8, zorder=3, label='FF (stacked)')

    # Total value labels on top of each stacked bar
    for xi, l, f in zip(x, LUT, FF):
        ax1.text(xi, l + f + 60, f'{l+f:,}',
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                 color=MC['black'])

    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, fontsize=10.5)
    ax1.set_ylabel('Total Resources  (LUT + FF)', fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax1.set_ylim(0, max(l+f for l, f in zip(LUT, FF)) * 1.18)

    # Accuracy line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, [a * 100 for a in ACC], 'o-',
             color=MC['red'], lw=2.5, ms=9, zorder=5, label='Accuracy (%)')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, color=MC['red'])
    ax2.tick_params(axis='y', colors=MC['red'])
    ax2.set_ylim(87, 101)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(MC['red'])
    ax2.spines['top'].set_visible(False)

    for xi, a in zip(x, ACC):
        ax2.annotate(f'{a*100:.1f}%', (xi, a * 100),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9, fontweight='bold', color=MC['red'])

    ax1.set_title('Total FPGA Resources vs Accuracy — All Models',
                  fontsize=14, fontweight='bold', pad=12)

    # Legend
    legend_handles = [
        mpatches.Patch(color=MC['orange'],  label='Baseline fc3'),
        mpatches.Patch(color=MC['lorange'], label='Baseline fc2+fc3'),
        mpatches.Patch(color=MC['blue'],    label='SR 1L (POL/SRL/SCE)'),
        mpatches.Patch(color=MC['teal'],    label='SR 2L (POL/SRL/SCE)'),
        mpatches.Patch(color=MC['gray'],    alpha=0.9, label='Solid = LUT'),
        mpatches.Patch(color=MC['gray'],    alpha=0.4, label='Faded = FF (stacked)'),
        plt.Line2D([0], [0], color=MC['red'], lw=2.5, marker='o',
                   ms=8, label='Accuracy (%)'),
    ]
    ax1.legend(handles=legend_handles, loc='upper left',
               fontsize=9, framealpha=0.9, ncol=2)

    fig.tight_layout()
    save(fig, '1_total_resources_vs_accuracy.png')


# ── Plot 2: Resource Comparison — all models vs baselines ─────────────────────
def plot_resources_comparison():
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    x = np.arange(len(LABELS))
    w = 0.55

    specs = [
        (LUT, 'LUT Count',       'Logic (Look-Up Tables)'),
        (FF,  'FF Count',        'Registers (Flip-Flops)'),
        (DSP, 'DSP Block Count', 'DSP Blocks'),
    ]

    for ax, (vals, ylabel, title) in zip(axes, specs):
        bars = ax.bar(x, vals, color=BAR_COLORS, width=w,
                      edgecolor='white', linewidth=0.8, zorder=3)

        # Baseline reference lines
        bl_fc3     = vals[0]   # Baseline fc3 value
        bl_fc2fc3  = vals[1]   # Baseline fc2+fc3 value
        ax.axhline(bl_fc3,    color=MC['orange'],  linestyle='--',
                   lw=1.6, zorder=2, alpha=0.9,
                   label=f'Baseline fc3 ({bl_fc3:,})')
        ax.axhline(bl_fc2fc3, color=MC['lorange'], linestyle=':',
                   lw=1.6, zorder=2, alpha=0.9,
                   label=f'Baseline fc2+fc3 ({bl_fc2fc3:,})')

        # Value labels
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + max(vals) * 0.012,
                    f'{v:,}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold')

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=5)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(LABELS, fontsize=10.5)

    # Shared color legend at top
    legend_handles = [
        mpatches.Patch(color=MC['orange'],  label='Baseline fc3'),
        mpatches.Patch(color=MC['lorange'], label='Baseline fc2+fc3'),
        mpatches.Patch(color=MC['blue'],    label='SR 1L (POL / SRL / SCE)'),
        mpatches.Patch(color=MC['teal'],    label='SR 2L (POL / SRL / SCE)'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               fontsize=10, framealpha=0.95,
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle('FPGA Resource Comparison — All Models vs Baselines',
                 fontsize=14, fontweight='bold', y=1.04)
    fig.tight_layout()
    save(fig, '2_resources_comparison.png')


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Saving plots to {OUT_DIR}')
    plot_area_vs_accuracy()
    plot_resources_comparison()
    print('Done.')
