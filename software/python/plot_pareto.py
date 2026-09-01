"""Phase E — Pareto charts for ICDV paper.

Two panels, all numbers traceable to SOTA_TABLE.md / PAPER_DATA.md (verified 2026-06-18):
  (A) Params vs Accuracy  — Chapman 4-superclass rhythm task (Table A).
  (B) Latency vs Accuracy — FPGA ECG accelerators (Table B).

Convention:
  - FILLED marker  = same task as ours (Chapman 4-superclass, rhythm-level) -> comparable.
  - HOLLOW/gray    = different dataset/task -> context only, NOT a direct accuracy ranking.
  - "Ours" plotted twice in (B): 8-PE channel-parallel + SIMD-20 (DSE front).

Output: results/figures/pareto_params_acc.{png,pdf}, pareto_latency_acc.{png,pdf},
        pareto_combined.png
"""
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIGDIR = os.path.join(os.path.dirname(__file__), "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)

OURS = "#c0392b"      # ours highlight
SAME = "#2c3e50"      # same-task competitor (filled)
OTHER = "#95a5a6"     # different task/dataset (context)

# ---------------------------------------------------------------------------
# Panel A: Params (log) vs Accuracy  — Chapman 4-superclass rhythm
# (params, acc, label, marker, filled?, ours?)
# ---------------------------------------------------------------------------
# (x, acc, label, marker, filled_same?, ours?, (label_x, label_y, ha))
panelA = [
    (640,    94.65, "Ours 1D-CNN INT8\n(640 params)", "*", True,  True,  (1.4e3, 93.95, "left")),
    (5.31e6, 98.73, "LightX3ECG (3-lead)",            "o", True,  False, (3.0e6, 99.15, "right")),
    (23e6,   95.08, "Bimodal CNN (Lead-II)",          "s", True,  False, (1.3e7, 95.15, "right")),
]

# ---------------------------------------------------------------------------
# Panel B: Latency (log, us) vs Accuracy  — FPGA accelerators
# (latency_us, acc, label, marker, same_task?, ours?)
# same_task=True only for Chapman 4-superclass rhythm (Ours + Liu).
# ---------------------------------------------------------------------------
# (x, acc, label, marker, same_task?, ours?, (label_x, label_y, ha))
panelB = [
    (52.16, 94.65, "Ours 8-PE (52 us)",    "*", True,  True,  (75, 95.6, "left")),
    (27.55, 94.65, "Ours SIMD-20 (28 us)", "P", True,  True,  (20, 93.5, "right")),
    (66.0,  92.95, "Liu 2023 (Chapman)",   "o", True,  False, (95, 92.6, "left")),
    (17e3,  94.20, "Carreras TCN",         "^", False, False, (2.3e4, 94.2, "left")),
    (1.37e3,92.07, "Xing SNN\n(MIT-BIH,inter-pat)", "D", False, False, (900, 91.4, "center")),
    (0.99,  99.82, "Wess MLP+PCA\n(MIT-BIH beat)",  "v", False, False, (1.4, 99.9, "left")),
    (17e6,  98.27, "Srivastava PNN\n(MIT-BIH beat)","<", False, False, (1.2e7, 98.3, "right")),
]


def scatter(ax, pts, xlabel):
    for x, y, lab, mk, same, ours, off in pts:
        if ours:
            fc, ec, z, sz = OURS, OURS, 5, 320
        elif same:
            fc, ec, z, sz = SAME, SAME, 4, 150
        else:
            fc, ec, z, sz = "none", OTHER, 3, 130
        ax.scatter(x, y, marker=mk, s=sz, facecolors=fc, edgecolors=ec,
                   linewidths=1.6, zorder=z)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.4)


def annotate(ax, pts):
    for x, y, lab, mk, same, ours, off in pts:
        col = OURS if ours else (SAME if same else OTHER)
        lx, ly, ha = off
        ax.annotate(lab, (x, y), xytext=(lx, ly), fontsize=8,
                    color=col, va="center", ha=ha)


# ---- Panel A standalone ----
figA, axA = plt.subplots(figsize=(6.2, 4.4))
scatter(axA, panelA, "Model parameters (log scale)")
annotate(axA, panelA)
axA.set_title("(A) Params vs Accuracy — Chapman 4-superclass (rhythm)", fontsize=10)
axA.set_xlim(200, 1e8)
axA.set_ylim(93.5, 99.5)
axA.annotate("~8,300x", (640, 94.65), xytext=(1.5e4, 96.7), fontsize=8,
             color=OURS, ha="center",
             arrowprops=dict(arrowstyle="->", color=OURS, lw=0.8))
figA.tight_layout()
figA.savefig(os.path.join(FIGDIR, "pareto_params_acc.png"), dpi=300)
figA.savefig(os.path.join(FIGDIR, "pareto_params_acc.pdf"))

# ---- Panel B standalone ----
figB, axB = plt.subplots(figsize=(6.6, 4.4))
scatter(axB, panelB, "Latency per inference (us, log scale)")
annotate(axB, panelB)
axB.set_title("(B) Latency vs Accuracy — FPGA ECG accelerators", fontsize=10)
axB.set_xlim(0.4, 1e8)
axB.set_ylim(91, 101)
legend = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor=OURS,
           markeredgecolor=OURS, markersize=15, label="Ours (Chapman, rhythm)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=SAME,
           markeredgecolor=SAME, markersize=10, label="Same task (Chapman rhythm)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
           markeredgecolor=OTHER, markersize=10, label="Other dataset/task (context)"),
]
axB.legend(handles=legend, fontsize=8, loc="lower right", framealpha=0.9)
figB.tight_layout()
figB.savefig(os.path.join(FIGDIR, "pareto_latency_acc.png"), dpi=300)
figB.savefig(os.path.join(FIGDIR, "pareto_latency_acc.pdf"))

# ---- Combined 1x2 ----
figC, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
scatter(a1, panelA, "Model parameters (log scale)")
annotate(a1, panelA)
a1.set_title("(A) Params vs Accuracy — Chapman 4-superclass", fontsize=10)
a1.set_xlim(200, 1e8); a1.set_ylim(93.5, 99.5)
a1.annotate("~8,300x smaller", (640, 94.65), xytext=(1.5e4, 96.7), fontsize=8,
            color=OURS, ha="center",
            arrowprops=dict(arrowstyle="->", color=OURS, lw=0.8))
scatter(a2, panelB, "Latency per inference (us, log scale)")
annotate(a2, panelB)
a2.set_title("(B) Latency vs Accuracy — FPGA ECG accelerators", fontsize=10)
a2.set_xlim(0.4, 1e8); a2.set_ylim(91, 101)
a2.legend(handles=legend, fontsize=8, loc="lower right", framealpha=0.9)
figC.tight_layout()
figC.savefig(os.path.join(FIGDIR, "pareto_combined.png"), dpi=300)

print("Saved to", FIGDIR)
for f in ("pareto_params_acc.png", "pareto_params_acc.pdf",
          "pareto_latency_acc.png", "pareto_latency_acc.pdf",
          "pareto_combined.png"):
    print("  ", f)
