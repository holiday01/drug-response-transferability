#!/usr/bin/env python3
"""Revision (2026-09-30): Figure 6 only (TCGA patient bridge), replacing the Fig 6
block of make_figures2.py (which also rewrites an obsolete Fig 5 and must not be rerun).
Fixes: panel A x-axis now shows the full RF bootstrap interval (cyclophosphamide lower
bound 0.040); panel B labels placed manually to avoid overlap; units stated on axes."""
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import RESULTS_DIR, FIG_DIR, FIG_TIFF_DIR
R = str(RESULTS_DIR)
OUT = str(FIG_DIR)
TIF = str(FIG_TIFF_DIR)
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "font.family": "DejaVu Sans"})
BLUE, RED, GREY, GREEN = "#2c6fbb", "#c0392b", "#8a8a8a", "#27893f"

t = pd.read_csv(f"{R}/R4_tcga_bridge_per_drug.csv"); t["rho"] = t["rho"].astype(float)
t = t.sort_values("patAUROC_mean")
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=True)
lo = np.array([float(c.split("-")[0]) for c in t["patAUROC_RF_ci"]])
hi = np.array([float(c.split("-")[1]) for c in t["patAUROC_RF_ci"]])
yv = np.arange(len(t)); rf = t["patAUROC_Random Forest"].values
ax[0].errorbar(rf, yv, xerr=[rf - lo, hi - rf], fmt="o", color=BLUE, ecolor=GREY, capsize=3, ms=6)
ax[0].axvline(0.5, color=RED, ls="--", lw=1)
ax[0].set_yticks(yv); ax[0].set_yticklabels([f"{d.title()} (n={n})" for d, n in zip(t["drug"], t["n_pat"])], fontsize=9)
ax[0].set_xlabel("Patient response AUROC\n(random forest, 95% CI)")
ax[0].set_title("A", loc="left", fontweight="bold")
ax[0].set_xlim(0.0, 0.85)
s = t[["rho", "patAUROC_mean", "drug"]].dropna()
ax[1].scatter(s["rho"], s["patAUROC_mean"], s=70, color=GREEN, edgecolor="k", linewidth=0.4, zorder=3)
OFF = {"FLUOROURACIL": (-9, -4, "right"), "DOCETAXEL": (0, -16, "center"), "GEMCITABINE": (0, 9, "center"),
       "PACLITAXEL": (-9, -3, "right"), "CYCLOPHOSPHAMIDE": (7, -3, "left")}
for _, r in s.iterrows():
    dx, dy, ha = OFF.get(r["drug"], (3, 3, "left"))
    ax[1].annotate(r["drug"].title(), (r["rho"], r["patAUROC_mean"]), fontsize=9, xytext=(dx, dy),
                   textcoords="offset points", ha=ha)
ax[1].axhline(0.5, color=RED, ls="--", lw=1); ax[1].set_xlim(-0.20, 0.90); ax[1].set_ylim(0.30, 0.66)
sp = stats.spearmanr(s["rho"], s["patAUROC_mean"])
ax[1].set_xlabel("Concordance $\\rho$"); ax[1].set_ylabel("Patient AUROC\n(mean of three models)")
ax[1].set_title("B", loc="left", fontweight="bold")
ax[1].text(0.97, 0.05, f"Spearman = {sp.statistic:+.2f} (n.s.)", va="bottom", ha="right", transform=ax[1].transAxes,
           fontsize=9, bbox=dict(boxstyle="round", fc="white", ec=GREY))
fig.savefig(f"{OUT}/fig6_tcga_bridge.pdf"); fig.savefig(f"{OUT}/fig6_tcga_bridge.png", dpi=300)
fig.savefig(f"{TIF}/fig6_tcga_bridge.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("wrote fig6_tcga_bridge")
