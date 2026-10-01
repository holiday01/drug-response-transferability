#!/usr/bin/env python3
"""Revision figures (2026-09-30): Fig 1, 2, 4, 5 regenerated from the independent
reference/test design with the official GDSC2 fitted AUC (results/rev*).
Fig 3 and Fig 6 are unchanged (make_figures.py / make_figures2.py).
Saves PDF + 300-dpi PNG + 600-dpi TIFF."""
import json
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats

from config import RESULTS_DIR, FIG_DIR, FIG_TIFF_DIR
R = str(RESULTS_DIR)
OUT = str(FIG_DIR)
TIF = str(FIG_TIFF_DIR)
plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "font.family": "DejaVu Sans",
})
BLUE, RED, GREY, GREEN, ORANGE, PURPLE = "#2c6fbb", "#c0392b", "#8a8a8a", "#27893f", "#e08a1e", "#7b4fa3"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    fig.savefig(f"{TIF}/{name}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("wrote", name)


def panel(a, s):
    a.set_title(s, loc="left", fontweight="bold")


def box(a, txt, x=0.03, y=0.88, ha="left"):
    a.text(x, y, txt, transform=a.transAxes, fontsize=9, ha=ha, va="top",
           bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9))


d = pd.read_csv(f"{R}/rev1_independent/drug_level_RF.csv")
S1 = json.load(open(f"{R}/rev1_independent/summary.json"))
EX = json.load(open(f"{R}/rev1_independent/extra_summary.json"))

# ============================================================ FIGURE 1
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.1), constrained_layout=True)
sc = ax[0].scatter(d.i, d.e, c=d.conc, cmap="viridis", s=44, edgecolor="k", linewidth=0.3, vmin=0, vmax=0.8)
ax[0].plot([0.5, 0.95], [0.5, 0.95], "--", color=GREY, lw=1)
ax[0].axhline(0.5, color=RED, ls=":", lw=0.9)
ax[0].set_xlabel("Internal AUROC (held-out CTRP)")
ax[0].set_ylabel("External AUROC (held-out GDSC2)")
ax[0].set_xlim(0.48, 0.95); ax[0].set_ylim(0.33, 0.95); panel(ax[0], "A")
cb = fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.03)
cb.set_label("Reference concordance $\\rho$", fontsize=9)
lr = stats.linregress(d.conc, d.e); xx = np.linspace(d.conc.min(), d.conc.max(), 50)
ax[1].scatter(d.conc, d.e, s=44, color=BLUE, edgecolor="k", linewidth=0.3, alpha=0.85)
ax[1].plot(xx, lr.slope * xx + lr.intercept, color=RED, lw=1.8)
ax[1].axhline(0.5, color=GREY, ls=":", lw=0.9); ax[1].axhline(0.7, color=GREEN, ls="--", lw=1.0)
ax[1].text(-0.06, 0.71, "0.70", color=GREEN, fontsize=9)
ax[1].set_xlabel("Reference concordance $\\rho$")
ax[1].set_ylabel("External AUROC (held-out GDSC2)"); panel(ax[1], "B")
m = S1["concordance_vs_external"]["fitted|Random Forest"]
box(ax[1], f"Spearman = {m['spearman']:.2f}\n95% CI {m['ci'][0]:.2f}–{m['ci'][1]:.2f}")
save(fig, "fig1_concordance_predicts_transfer")

# ============================================================ FIGURE 2
fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.8), constrained_layout=True)
t = EX["tertiles"]; keys = ["low", "mid", "high"]; xp = np.arange(3); w = 0.36
ax[0].bar(xp - w / 2, [t[k]["median_int"] for k in keys], w, color=GREY, label="Internal")
ax[0].bar(xp + w / 2, [t[k]["median_ext"] for k in keys], w, color=BLUE, label="External")
for i, k in enumerate(keys):
    ax[0].text(i, max(t[k]["median_int"], t[k]["median_ext"]) + 0.012, f"drop {t[k]['drop_pts']:.1f}",
               ha="center", fontsize=8.5, color=RED)
ax[0].set_xticks(xp); ax[0].set_xticklabels([f"{k}\n$\\rho${chr(0x2248)}{t[k]['median_conc']:.2f}" for k in keys])
ax[0].set_ylim(0.4, 1.02); ax[0].axhline(0.5, color=RED, ls=":", lw=0.8)
ax[0].set_ylabel("Median AUROC"); ax[0].legend(loc="upper left", fontsize=7.5, ncol=2, columnspacing=0.6, handlelength=1.0, frameon=False); panel(ax[0], "A")
fpr, tpr, _ = roc_curve(d.ok, d.conc)
ax[1].plot(fpr, tpr, color=BLUE, lw=2.2); ax[1].plot([0, 1], [0, 1], "--", color=GREY, lw=1)
ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate"); panel(ax[1], "B")
box(ax[1], f"ROC-AUC = {roc_auc_score(d.ok, d.conc):.2f}\nLODO = {EX['lodo']['roc_auc']:.2f}", x=0.35, y=0.3)
hi, lo = d[d.conc >= 0.5].e.values, d[d.conc < 0.5].e.values
bp = ax[2].boxplot([lo, hi], widths=0.55, patch_artist=True,
                   tick_labels=[f"$\\rho$<0.5\n(n={len(lo)})", f"$\\rho\\geq$0.5\n(n={len(hi)})"])
for p, c in zip(bp["boxes"], [RED, GREEN]): p.set_facecolor(c); p.set_alpha(0.35)
for med in bp["medians"]: med.set_color("k")
rng = np.random.default_rng(0)
ax[2].scatter(rng.normal(1, 0.05, len(lo)), lo, s=16, color=RED, alpha=0.6, zorder=3)
ax[2].scatter(rng.normal(2, 0.05, len(hi)), hi, s=16, color=GREEN, alpha=0.6, zorder=3)
ax[2].axhline(0.5, color=GREY, ls=":", lw=0.8); ax[2].axhline(0.7, color=GREEN, ls="--", lw=0.8)
ax[2].set_ylabel("External AUROC"); panel(ax[2], "C")
save(fig, "fig2_screening")

# ============================================================ FIGURE 4 (label strategies)
S3 = json.load(open(f"{R}/rev3_consensus/summary.json"))
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1]})
sch = [("S", "CTRP only", GREY), ("G", "GDSC2 only", PURPLE), ("A", "Averaged", BLUE),
       ("C", "Consensus", GREEN), ("D", "Size-matched", ORANGE), ("X", "CTRP extremes", "#b5a642")]
groups = [("Logistic Regression", "gdsc2", "LR\nGDSC2"), ("Logistic Regression", "prism", "LR\nPRISM"),
          ("Random Forest", "gdsc2", "RF\nGDSC2"), ("Random Forest", "prism", "RF\nPRISM")]
xp = np.arange(len(groups)); w = 0.13
for j, (k, lab, col) in enumerate(sch):
    ax[0].bar(xp + (j - 2.5) * w, [S3[f"{m}|{tg}"]["median_auroc"][k] for m, tg, _ in groups], w,
              color=col, label=lab, edgecolor="k", linewidth=0.3)
ax[0].set_xticks(xp); ax[0].set_xticklabels([g[2] for g in groups]); ax[0].set_ylim(0.55, 0.76)
ax[0].set_ylabel("Median AUROC on held-out lines")
ax[0].legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), columnspacing=0.8, handlelength=1.2)
panel(ax[0], "A")
rows = []
for m, mk in [("Logistic Regression", "LR"), ("Random Forest", "RF")]:
    for tg in ["gdsc2", "prism"]:
        for k, lab in [("S", "CTRP only"), ("G", "GDSC2 only"), ("A", "averaged")]:
            e = S3[f"{m}|{tg}"][f"C_minus_{k}"]
            rows.append((f"{mk} {tg.upper()}: vs {lab}", e["median_pts"], *e["ci_pts"]))
yv = np.arange(len(rows))[::-1]
for y, (lab, mdn, l, h) in zip(yv, rows):
    col = GREEN if l > 0 else (RED if h < 0 else GREY)
    ax[1].errorbar(mdn, y, xerr=[[mdn - l], [h - mdn]], fmt="o", color=col, ecolor=col, capsize=2.5, ms=4.5)
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_yticks(yv); ax[1].set_yticklabels([r[0] for r in rows], fontsize=8)
ax[1].set_xlabel("Consensus $-$ comparator\n($\\Delta$AUROC, pts; 95% CI)"); panel(ax[1], "B")
save(fig, "fig4_consensus")

# ============================================================ FIGURE 5 (2 x 2)
fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.6), constrained_layout=True)
lo3 = pd.read_csv(f"{R}/rev2_three_assay/loo_random.csv")
S2 = json.load(open(f"{R}/rev2_three_assay/summary.json"))["Random Forest"]
for H, col in [("CTRP", BLUE), ("GDSC2", PURPLE), ("PRISM", ORANGE)]:
    x = lo3[lo3.held_out == H]
    ax[0, 0].scatter(x.score, x.transfer, s=30, color=col, edgecolor="k", linewidth=0.3, alpha=0.85,
                     label=f"{H} held out ($\\rho_s$={S2[H]['spearman']:.2f})")
ax[0, 0].axhline(0.5, color=GREY, ls=":", lw=0.9)
ax[0, 0].set_xlabel("Concordance of the other two assays")
ax[0, 0].set_ylabel("Transfer into held-out assay\n(external AUROC)")
ax[0, 0].set_ylim(0.37, 1.08)
ax[0, 0].legend(fontsize=8, loc="upper left", handletextpad=0.2, borderpad=0.3, framealpha=0.9); panel(ax[0, 0], "A")
# B: single reference set per drug (one draw, one split) -- rev1b_refsize_single
rep = pd.read_csv(f"{R}/rev1b_refsize_single/replicates.csv")
fr = pd.read_csv(f"{R}/rev1b_refsize_single/full_reference_per_split.csv")
nref = int(S1["reference_size_sweep_RF_fitted"]["full_reference"]["median_n_ref"])
groups = [rep[rep.k == k].spearman.to_numpy() for k in (20, 50, 100)] + [fr.spearman.to_numpy()]
bp = ax[0, 1].boxplot(groups, positions=range(4), widths=0.5, whis=(2.5, 97.5), showfliers=False,
                      patch_artist=True, medianprops=dict(color="k", lw=1.4))
for b, c in zip(bp["boxes"], [BLUE, BLUE, BLUE, GREEN]):
    b.set_facecolor(c); b.set_alpha(0.45)
for i, g in enumerate(groups):
    ax[0, 1].text(i, np.percentile(g, 97.5) + 0.03, f"{np.median(g):.2f}", ha="center", fontsize=8.5)
ax[0, 1].set_xticks(range(4)); ax[0, 1].set_xticklabels(["20", "50", "100", f"all\n(~{nref})"])
ax[0, 1].set_ylim(0.1, 1.0); ax[0, 1].set_xlabel("Reference cell lines with both assays")
ax[0, 1].set_ylabel("Spearman (concordance,\nexternal AUROC)"); panel(ax[0, 1], "B")
cal = S1["transfer_rate_by_bin_RF_fitted"]
labs = ["≤0.20", "0.20–0.35", "0.35–0.50", "0.50–0.65", ">0.65"]
rate = [v["rate"] for v in cal.values()]; ci = [v["wilson95"] for v in cal.values()]
kn = [f"{v['k']}/{v['n']}" for v in cal.values()]
xb = np.arange(len(rate))
ax[1, 0].bar(xb, rate, color=GREEN, alpha=0.75, edgecolor="k", linewidth=0.4)
ax[1, 0].errorbar(xb, rate, yerr=[np.array(rate) - [c[0] for c in ci], [c[1] for c in ci] - np.array(rate)],
                  fmt="none", ecolor="k", capsize=3, lw=0.9)
for i, s in enumerate(kn):
    ax[1, 0].text(i, -0.1, s, ha="center", fontsize=8)
ax[1, 0].set_xticks(xb); ax[1, 0].set_xticklabels(labs, rotation=25, fontsize=8)
ax[1, 0].set_ylim(-0.16, 1.08); ax[1, 0].axhline(0, color="k", lw=0.6)
ax[1, 0].set_ylabel("Transferable drugs\n(external AUROC ≥ 0.70)"); ax[1, 0].set_xlabel("Reference concordance bin")
panel(ax[1, 0], "C")
bd = pd.read_csv(f"{R}/R6_bidirectional_per_drug.csv"); bd["rho"] = bd["rho"].astype(float)
s = bd[["rho", "ext_Random Forest"]].dropna(); lr = stats.linregress(s.rho, s["ext_Random Forest"])
xx = np.linspace(s.rho.min(), s.rho.max(), 40)
ax[1, 1].scatter(s.rho, s["ext_Random Forest"], s=30, color=PURPLE, edgecolor="k", linewidth=0.3, alpha=0.85)
ax[1, 1].plot(xx, lr.slope * xx + lr.intercept, color=RED, lw=1.6); ax[1, 1].axhline(0.5, color=GREY, ls=":", lw=0.9)
ax[1, 1].set_xlabel("Concordance $\\rho$"); ax[1, 1].set_ylabel("External AUROC (CTRP)")
box(ax[1, 1], f"Reverse direction\nSpearman = {stats.spearmanr(s.rho, s['ext_Random Forest']).statistic:.2f}")
panel(ax[1, 1], "D")
save(fig, "fig5_robustness")
