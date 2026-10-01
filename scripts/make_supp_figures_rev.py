#!/usr/bin/env python3
"""Revision supplementary figures (2026-09-30): S1, S2, S9 regenerated from the
REV-1 (reference/test, fitted GDSC2) and REV-2 (three-assay, five splits) results.
S3-S8 are unchanged (make_supp_figures.py, original design)."""
import numpy as np, pandas as pd, json
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import RESULTS_DIR, SUPP_FIG_DIR
R = str(RESULTS_DIR)
OUT = str(SUPP_FIG_DIR)
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "font.family": "DejaVu Sans"})
BLUE, RED, GREY, GREEN, ORANGE, PURPLE, TEAL, PINK = "#2c6fbb", "#c0392b", "#8a8a8a", "#27893f", "#e08a1e", "#7b4fa3", "#1b9e9e", "#c94f9a"
PAIRCOL = {"CTRP->GDSC2": BLUE, "CTRP->PRISM": ORANGE, "GDSC2->PRISM": GREEN,
           "GDSC2->CTRP": PURPLE, "PRISM->CTRP": TEAL, "PRISM->GDSC2": PINK}
HCOL = {"CTRP": BLUE, "GDSC2": PURPLE, "PRISM": ORANGE}


def save(fig, n):
    fig.savefig(f"{OUT}/{n}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{n}.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", n)


def fit(ax, x, y):
    sl, ic, *_ = stats.linregress(x, y); xx = np.linspace(np.min(x), np.max(x), 40)
    ax.plot(xx, sl * xx + ic, color=RED, lw=1.6, zorder=2)


pp = pd.read_csv(f"{R}/rev2_three_assay/per_pair_split.csv")
tp = pp.groupby(["drug", "src", "dst"]).agg(rho=("rho", "first"), ext=("ext_Random Forest", "mean")).reset_index()
tp["pair"] = tp.src + "->" + tp.dst

# ============ S1: six ordered pairs
fig, ax = plt.subplots(2, 3, figsize=(7.4, 5.0), constrained_layout=True)
for a, pr in zip(ax.ravel(), ["CTRP->GDSC2", "GDSC2->CTRP", "CTRP->PRISM", "PRISM->CTRP", "GDSC2->PRISM", "PRISM->GDSC2"]):
    s = tp[tp.pair == pr].dropna(subset=["rho", "ext"])
    a.scatter(s.rho, s.ext, s=26, color=PAIRCOL[pr], edgecolor="k", linewidth=0.25, alpha=0.85)
    fit(a, s.rho.values, s.ext.values); a.axhline(0.5, color=GREY, ls=":", lw=0.8)
    a.set_title(pr.replace("->", " $\\to$ "), fontsize=9.5, fontweight="bold")
    a.text(0.04, 0.95, f"$\\rho_s$={stats.spearmanr(s.rho, s.ext).statistic:.2f}\nn={len(s)}", transform=a.transAxes,
           fontsize=8.5, va="top", bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=.9))
    a.set_xlabel("Concordance"); a.set_ylabel("External AUROC")
save(fig, "figS1_threeassay_grid")

# ============ S2: pooled pairs + LOAO (logistic regression)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
for pr in PAIRCOL:
    s = tp[tp.pair == pr]
    ax[0].scatter(s.rho, s.ext, s=20, color=PAIRCOL[pr], edgecolor="none", alpha=0.7, label=pr.replace("->", "$\\to$"))
s = tp.dropna(subset=["rho", "ext"]); fit(ax[0], s.rho.values, s.ext.values)
ax[0].axhline(0.5, color=GREY, ls=":", lw=0.8); ax[0].set_xlabel("Cross-assay concordance $\\rho$"); ax[0].set_ylabel("External AUROC")
ax[0].set_title("A", loc="left", fontweight="bold"); ax[0].legend(fontsize=8, ncol=2, loc="lower right", framealpha=.9, handletextpad=0.2, columnspacing=0.6)
ax[0].text(0.04, 0.95, f"pooled $\\rho_s$={stats.spearmanr(s.rho, s.ext).statistic:.2f}\nn={len(s)}", transform=ax[0].transAxes,
           fontsize=8.5, va="top", bbox=dict(boxstyle="round", fc="white", ec=GREY))
lo = pd.read_csv(f"{R}/rev2_three_assay/loo_logistic.csv")
S2 = json.load(open(f"{R}/rev2_three_assay/summary.json"))["Logistic Regression"]
for H, c in HCOL.items():
    x = lo[lo.held_out == H]
    ax[1].scatter(x.score, x.transfer, s=28, color=c, edgecolor="k", linewidth=0.3, alpha=0.85,
                  label=f"{H} held out ($\\rho_s$={S2[H]['spearman']:.2f})")
ax[1].axhline(0.5, color=GREY, ls=":", lw=0.8); ax[1].set_ylim(0.35, 1.05)
ax[1].set_xlabel("Concordance of the other two assays"); ax[1].set_ylabel("Transfer into held-out assay")
ax[1].set_title("B", loc="left", fontweight="bold"); ax[1].legend(fontsize=8.5, loc="upper left", framealpha=.9)
save(fig, "figS2_pooled_loo")

# ============ S9: per-drug internal vs external, five models (reference/test design)
df = pd.read_csv(f"{R}/rev1_independent/per_drug_split.csv")
MODS = ["Logistic Regression", "Random Forest", "XGBoost", "SVM (RBF)", "DNN (MLP)"]
fig, ax = plt.subplots(2, 3, figsize=(7.4, 5.0), constrained_layout=True)
for a, m in zip(ax.ravel(), MODS):
    g = df.dropna(subset=[f"ext_fitted_{m}"]).groupby("drug").agg(i=(f"int_{m}", "mean"), e=(f"ext_fitted_{m}", "mean"), c=("conc_fitted", "mean"))
    a.scatter(g.i, g.e, c=g.c, cmap="viridis", s=26, edgecolor="k", linewidth=0.25, vmin=0, vmax=0.8)
    a.plot([0.5, 0.95], [0.5, 0.95], "--", color=GREY, lw=0.9); a.axhline(0.5, color=RED, ls=":", lw=0.8)
    a.set_title(m, fontsize=9.5, fontweight="bold"); a.set_xlabel("Internal AUROC"); a.set_ylabel("External AUROC")
    a.set_xlim(0.48, 0.95); a.set_ylim(0.33, 0.95)
ax.ravel()[5].axis("off")
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 0.8)); sm.set_array([])
cb = fig.colorbar(sm, ax=ax.ravel()[5], fraction=0.5, pad=0.05); cb.set_label("Reference concordance $\\rho$", fontsize=9)
save(fig, "figS9_per_model")
