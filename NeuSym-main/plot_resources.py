"""
plot_resources.py — Plot HLS resource utilization for all symbolic experiments
vs baseline fc3 and fc2+fc3.
"""

import matplotlib
matplotlib.use("Agg")   # headless server — no display needed
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────

models = [
    "Baseline\nfc3", "Baseline\nfc2+fc3",
    "SR-1L-POL", "SR-1L-SCE", "SR-1L-SRL",
    "SR-2L-POL", "SR-2L-SCE", "SR-2L-SRL",
]

dsp = [5,   5,    7,  10,  7,   2,  20,  7]
lut = [943, 2035, 1369, 1648, 1429, 661, 4280, 1490]
ff  = [498, 4813, 683,  853,  741,  408, 2719,  916]

# ── Plot ──────────────────────────────────────────────────────────────────────

x      = np.arange(len(models))
width  = 0.25
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("HLS Resource Utilization — Symbolic Heads vs Baselines\n(XC7Z020, 100 MHz)",
             fontsize=13, fontweight="bold")

# first 2 bars are baselines, rest are symbolic
bar_colors = ["#E05C5C", "#C0392B"] + ["#4C72B0"] * 3 + ["#55A868"] * 3

def _bar_group(ax, values, ylabel, title):
    bars = ax.bar(x, values, width=0.5, color=bar_colors, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_ylim(0, max(values) * 1.25)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E05C5C", label="Baseline fc3"),
        Patch(facecolor="#C0392B", label="Baseline fc2+fc3"),
        Patch(facecolor="#4C72B0", label="1L symbolic"),
        Patch(facecolor="#55A868", label="2L symbolic"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

_bar_group(axes[0], dsp, "DSP Blocks", "DSP Utilization")
_bar_group(axes[1], lut, "LUTs",       "LUT Utilization")
_bar_group(axes[2], ff,  "Flip-Flops", "FF Utilization")

plt.tight_layout()
out = "resource_utilization.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")


# ── Figure 2: Total normalized utilization (stacked % of available) ───────────

DSP_AVAIL = 220
LUT_AVAIL = 53200
FF_AVAIL  = 106400

dsp_pct = [v / DSP_AVAIL * 100 for v in dsp]
lut_pct = [v / LUT_AVAIL * 100 for v in lut]
ff_pct  = [v / FF_AVAIL  * 100 for v in ff]

fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.suptitle("Total FPGA Resource Utilization (% of Available)\n(XC7Z020, 100 MHz)",
              fontsize=13, fontweight="bold")

from matplotlib.patches import Patch

b1 = ax2.bar(x, dsp_pct, width=0.5, color="#4C72B0", zorder=3, label="DSP")
b2 = ax2.bar(x, lut_pct, width=0.5, color="#DD8452", zorder=3, label="LUT",
             bottom=dsp_pct)
b3 = ax2.bar(x, ff_pct,  width=0.5, color="#55A868", zorder=3, label="FF",
             bottom=[d + l for d, l in zip(dsp_pct, lut_pct)])

# Vertical lines separating baselines from symbolic
ax2.axvline(1.35, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax2.text(1.45, max([d+l+f for d,l,f in zip(dsp_pct,lut_pct,ff_pct)]) * 0.95,
         "← Baselines   Symbolic →", fontsize=9, color="gray")

ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
ax2.set_ylabel("% of Available Resources", fontsize=10)
ax2.set_title("Stacked Resource Utilization per Design", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.4, zorder=0)
ax2.set_ylim(0, max([d+l+f for d,l,f in zip(dsp_pct,lut_pct,ff_pct)]) * 1.3)

plt.tight_layout()
out2 = "resource_utilization_total.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")
