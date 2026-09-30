#!/usr/bin/env python3
"""Publication figures for the transferability paper. All data from
benchmark/results/*.csv (real, leakage-free, seed 42). Saves PDF + 300-dpi PNG.
Fonts are sized so that, placed at \\textwidth (~6.3 in), effective sizes stay
>= ~8.5 pt (figure-checker requirement)."""
import numpy as np, pandas as pd, json
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats

from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
R = str(RESULTS_DIR)
OUT = str(FIG_DIR)
# figures are placed at \textwidth (~6.27in); native width kept near that so the
# downscale is small and every printed font stays >= 8pt (figure-checker standard).
plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "font.family": "DejaVu Sans",
})
BLUE, RED, GREY, GREEN, ORANGE = "#2c6fbb", "#c0392b", "#8a8a8a", "#27893f", "#e08a1e"
PRIMARY = "Random Forest"

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)

ext = pd.read_csv(f"{R}/per_drug_external.csv")
ext = ext[ext["n_ext"] > 0].copy()
ext["rho"] = ext["spearman_ctrp_gdsc2"].astype(float)
iP, eP = f"int_{PRIMARY}", f"ext_{PRIMARY}"

# ============================================================ FIGURE 1 (2 panels)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.1), constrained_layout=True)
sc = ax[0].scatter(ext[iP], ext[eP], c=ext["rho"], cmap="viridis",
                   s=44, edgecolor="k", linewidth=0.3, vmin=0, vmax=0.8)
ax[0].plot([0.5, 0.95], [0.5, 0.95], "--", color=GREY, lw=1)
ax[0].axhline(0.5, color=RED, ls=":", lw=0.9)
ax[0].set_xlabel("Internal AUROC (held-out CTRP)")
ax[0].set_ylabel("External AUROC (independent GDSC2)")
ax[0].set_title("A", loc="left", fontweight="bold")
ax[0].set_xlim(0.5, 0.95); ax[0].set_ylim(0.4, 0.95)
cb = fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.03)
cb.set_label("CTRP–GDSC2 concordance $\\rho$", fontsize=9)
x, y = ext["rho"].to_numpy(), ext[eP].to_numpy()
sl, ic, rv, pv, _ = stats.linregress(x, y)
sp = stats.spearmanr(x, y).statistic
xx = np.linspace(x.min(), x.max(), 50)
ax[1].scatter(x, y, s=44, color=BLUE, edgecolor="k", linewidth=0.3, alpha=0.85)
ax[1].plot(xx, sl*xx+ic, color=RED, lw=1.8)
ax[1].axhline(0.5, color=GREY, ls=":", lw=0.9)
ax[1].axhline(0.7, color=GREEN, ls="--", lw=1.0)
ax[1].text(0.0, 0.715, "deployable (0.70)", color=GREEN, fontsize=9.5)
ax[1].set_xlabel("CTRP–GDSC2 concordance $\\rho$")
ax[1].set_ylabel("External AUROC (independent GDSC2)")
ax[1].set_title("B", loc="left", fontweight="bold")
ax[1].text(0.03, 0.88, f"Spearman = {sp:.2f}\n$R^2$ = {rv**2:.2f}, $P$ < 10$^{{-14}}$",
           transform=ax[1].transAxes, fontsize=9.5,
           bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9))
save(fig, "fig1_concordance_predicts_transfer")

# ============================================================ FIGURE 2 (3 panels)
fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.7), constrained_layout=True)
t = pd.read_csv(f"{R}/R2_concordance_tertiles.csv")
order = {"low": 0, "mid": 1, "high": 2}
t = t.sort_values("tertile", key=lambda s: s.map(order))
xpos = np.arange(len(t)); w = 0.36
ax[0].bar(xpos-w/2, t["median_internal"], w, color=GREY, label="Internal (CTRP)")
ax[0].bar(xpos+w/2, t["median_external"], w, color=BLUE, label="External (GDSC2)")
for i, (_, r) in enumerate(t.iterrows()):
    ax[0].text(i, max(r.median_internal, r.median_external)+0.012,
               f"drop {r.median_drop_pts:.1f}", ha="center", fontsize=9.5, color=RED)
ax[0].set_xticks(xpos); ax[0].set_xticklabels([f"{s}\n$\\rho${chr(0x2248)}{r:.2f}"
    for s, r in zip(t["tertile"], t["rho_median"])])
ax[0].set_ylim(0.4, 0.95); ax[0].axhline(0.5, color=RED, ls=":", lw=0.8)
ax[0].set_ylabel("Median AUROC"); ax[0].legend(loc="lower right")
ax[0].set_title("A", loc="left", fontweight="bold")
label = (ext[eP] >= 0.70).astype(int).to_numpy(); score = ext["rho"].to_numpy()
fpr, tpr, _ = roc_curve(label, score); auc = roc_auc_score(label, score)
ax[1].plot(fpr, tpr, color=BLUE, lw=2.2)
ax[1].plot([0, 1], [0, 1], "--", color=GREY, lw=1)
ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
ax[1].set_title("B", loc="left", fontweight="bold")
ax[1].text(0.32, 0.12, f"ROC-AUC = {auc:.2f}", fontsize=9,
           bbox=dict(boxstyle="round", fc="white", ec=GREY))
hi = ext[ext["rho"] >= 0.5][eP].to_numpy()
lo = ext[ext["rho"] < 0.5][eP].to_numpy()
bp = ax[2].boxplot([lo, hi], widths=0.55, patch_artist=True,
                   tick_labels=[f"discarded\n$\\rho$<0.5 (n={len(lo)})",
                                f"deployed\n$\\rho$$\\geq$0.5 (n={len(hi)})"])
for patch, c in zip(bp["boxes"], [RED, GREEN]):
    patch.set_facecolor(c); patch.set_alpha(0.35)
for med in bp["medians"]: med.set_color("k")
ax[2].scatter(np.random.default_rng(0).normal(1, 0.05, len(lo)), lo, s=16, color=RED, alpha=0.6, zorder=3)
ax[2].scatter(np.random.default_rng(1).normal(2, 0.05, len(hi)), hi, s=16, color=GREEN, alpha=0.6, zorder=3)
ax[2].axhline(0.5, color=GREY, ls=":", lw=0.8); ax[2].axhline(0.7, color=GREEN, ls="--", lw=0.8)
ax[2].set_ylabel("External AUROC")
ax[2].set_title("C", loc="left", fontweight="bold")
save(fig, "fig2_screening")

# ============================================================ FIGURE 3 (2 panels)
det = pd.read_csv(f"{R}/R5_determinants_per_drug.csv")
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.1), constrained_layout=True)
s = det[["rho", "gdsc_dynrange"]].dropna()
sp = stats.spearmanr(s["rho"], s["gdsc_dynrange"]).statistic
ax[0].scatter(s["gdsc_dynrange"], s["rho"], s=42, color=ORANGE, edgecolor="k", linewidth=0.3, alpha=0.85)
ax[0].set_xlabel("Phenotype dynamic range (GDSC2 s.d.)")
ax[0].set_ylabel("CTRP–GDSC2 concordance $\\rho$")
ax[0].set_title("A", loc="left", fontweight="bold")
ax[0].text(0.04, 0.90, f"Spearman = {sp:.2f}", transform=ax[0].transAxes, fontsize=9.5,
           bbox=dict(boxstyle="round", fc="white", ec=GREY))
feats = [("gdsc_dynrange", "Dynamic range\n(GDSC2)"),
         ("ctrp_dynrange", "Dynamic range\n(CTRP)"),
         ("int_RF", "Internal\ndifficulty"),
         ("lineage_entropy", "Lineage\nspread")]
vals, labs, cols = [], [], []
for k, lab in feats:
    ss = det[["rho", k]].dropna()
    r = stats.spearmanr(ss["rho"], ss[k]).statistic
    vals.append(r); labs.append(lab); cols.append(GREEN if r > 0 else RED)
yb = np.arange(len(vals))
ax[1].barh(yb, vals, color=cols, alpha=0.75, edgecolor="k", linewidth=0.4)
ax[1].set_yticks(yb); ax[1].set_yticklabels(labs)
ax[1].axvline(0, color="k", lw=0.8); ax[1].set_xlim(-0.5, 0.9)
ax[1].set_xlabel("Spearman correlation with $\\rho$")
ax[1].set_title("B", loc="left", fontweight="bold")
ax[1].invert_yaxis()
save(fig, "fig3_determinants")

# ============================================================ FIGURE 4 (2 panels)
cons = pd.read_csv(f"{R}/R3_consensus_per_drug.csv")
summ = json.load(open(f"{R}/R3_consensus_summary.json"))
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.1), constrained_layout=True)
mods = ["Random Forest", "Logistic Regression"]
schemes = [("S", "Single-assay", GREY), ("C", "Consensus", GREEN), ("D", "Size-matched random", ORANGE)]
xpos = np.arange(len(mods)); w = 0.25
for j, (sc_, lab, col) in enumerate(schemes):
    ys = [summ[f"{m}|cons"][f"median_{sc_}"] for m in mods]
    ax[0].bar(xpos + (j-1)*w, ys, w, color=col, alpha=0.85, label=lab, edgecolor="k", linewidth=0.3)
ax[0].set_xticks(xpos); ax[0].set_xticklabels(["Random\nForest", "Logistic\nRegression"])
ax[0].set_ylim(0.5, 0.99); ax[0].set_ylabel("Median AUROC (reproducible target)")
ax[0].legend(loc="lower left", ncol=1, fontsize=9, framealpha=0.9)
ax[0].set_title("A", loc="left", fontweight="bold")
lr = summ["Logistic Regression|cons"]
ax[0].annotate(f"+{lr['delta_C_minus_S_pts']:.1f} pts, $P$<10$^{{-7}}$",
               xy=(1, lr["median_C"]), xytext=(1.0, 0.955), fontsize=9.5, color=GREEN,
               ha="center", arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9))
d = (cons["C_Logistic Regression|cons"] - cons["S_Logistic Regression|cons"]).dropna().sort_values().to_numpy()*100
colors = [GREEN if v > 0 else RED for v in d]
ax[1].bar(np.arange(len(d)), d, color=colors, width=1.0)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].axhline(np.median(d), color=BLUE, ls="--", lw=1.2, label=f"median +{np.median(d):.1f} pts")
ax[1].set_xlabel(f"Drugs (sorted, n={len(d)})")
ax[1].set_ylabel("$\\Delta$ AUROC, consensus $-$ single (pts)")
ax[1].set_title("B", loc="left", fontweight="bold")
ax[1].legend(loc="upper left")
save(fig, "fig4_consensus")
print("\nAll figures written to", OUT)
