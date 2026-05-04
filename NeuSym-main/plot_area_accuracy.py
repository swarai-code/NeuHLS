"""
plot_area_accuracy.py — Area (LUT) as bars, Accuracy as line overlay.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────

models = [
    "Baseline\nfc3", "Baseline\nfc2+fc3",
    "SR-1L-POL", "SR-1L-SCE", "SR-1L-SRL",
    "SR-2L-POL", "SR-2L-SCE", "SR-2L-SRL",
]

lut      = [943,  2035, 1369, 1648, 1429,  661, 4280, 1490]
# Fine-tuned accuracy (baselines use original accuracy as reference)
accuracy = [95.35, 95.35, 95.08, 95.07, 94.90, 92.53, 92.75, 93.38]

bar_colors = ["#E05C5C", "#C0392B"] + ["#4C72B0"] * 3 + ["#55A868"] * 3

# ── Plot ──────────────────────────────────────────────────────────────────────

x = np.arange(len(models))

fig, ax1 = plt.subplots(figsize=(13, 6))

# Bar chart — LUT (left axis)
bars = ax1.bar(x, lut, width=0.5, color=bar_colors, zorder=3, alpha=0.85)
ax1.set_xlabel("Model", fontsize=11)
ax1.set_ylabel("LUT Utilization", fontsize=11, color="black")
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
ax1.set_ylim(0, max(lut) * 1.35)
ax1.grid(axis="y", alpha=0.3, zorder=0)

# Value labels on bars
for bar, val in zip(bars, lut):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
             str(val), ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Line chart — Accuracy (right axis)
ax2 = ax1.twinx()
ax2.plot(x, accuracy, color="black", marker="o", linewidth=2,
         markersize=7, zorder=5, label="Test Accuracy")
ax2.set_ylabel("Test Accuracy (%)", fontsize=11, color="black")
ax2.set_ylim(88, 98)

# Accuracy value labels
for xi, acc in zip(x, accuracy):
    ax2.annotate(f"{acc:.1f}%", (xi, acc),
                 textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=8.5, color="black", fontweight="bold")

# Baseline accuracy reference line
ax2.axhline(95.35, color="gray", linestyle="--", linewidth=1.2, alpha=0.6,
            label="Baseline accuracy (95.35%)")

# Title and legend
fig.suptitle("LUT Area vs Test Accuracy (After Fine-Tuning) — Symbolic Heads vs Baselines\n(XC7Z020, 100 MHz)",
             fontsize=12, fontweight="bold")

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor="#E05C5C", label="Baseline fc3"),
    Patch(facecolor="#C0392B", label="Baseline fc2+fc3"),
    Patch(facecolor="#4C72B0", label="1L symbolic"),
    Patch(facecolor="#55A868", label="2L symbolic"),
    Line2D([0], [0], color="black", marker="o", markersize=6, label="Test accuracy"),
    Line2D([0], [0], color="gray",  linestyle="--", linewidth=1.2, label="Baseline accuracy"),
]
ax1.legend(handles=legend_elements, fontsize=8.5, loc="upper left")

plt.tight_layout()
out = "area_vs_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
