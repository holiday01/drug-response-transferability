#!/usr/bin/env python3
"""
REV-1 analysis: summarise results/rev1_independent/per_drug_split.csv.

Drug-level statistics use each drug's mean over splits (one observation per drug);
uncertainty is a drug-level bootstrap. Per-split correlations are also reported to
show split-to-split variability. Partial rank correlation uses the conventional
definition: Pearson correlation of residuals of the rank-transformed variables.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

SEED = 42
from config import REV1_DIR
D = REV1_DIR
df = pd.read_csv(D / "per_drug_split.csv")
MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "SVM (RBF)", "DNN (MLP)"]
SUMS = ["fitted", "trapz", "legacy"]
THR_AUROC, THR_CONC = 0.70, 0.50
rng = np.random.default_rng(SEED)
out = {}


def boot_spearman(x, y, n=5000):
    x, y = np.asarray(x), np.asarray(y); k = len(x); bs = []
    for _ in range(n):
        i = rng.integers(0, k, k); bs.append(stats.spearmanr(x[i], y[i]).statistic)
    return [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


def partial_rank(x, y, Z):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Zr = np.column_stack([np.ones(len(x))] + [stats.rankdata(z) for z in Z])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1]); dof = len(x) - 2 - (Zr.shape[1] - 1)
    t = r * np.sqrt(dof / (1 - r ** 2)); p = float(2 * stats.t.sf(abs(t), dof))
    return round(r, 4), p, dof


def wilson(k, n, z=1.96):
    if n == 0: return [np.nan, np.nan]
    p = k / n; d = 1 + z ** 2 / n; c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return [round(c - h, 3), round(c + h, 3)]


out["n_drugs"] = int(df.drug.nunique()); out["n_splits"] = int(df.split.nunique())
out["test_lines_median"] = float(df.n_T.median())

# ---------------- main: concordance (reference only) vs external AUROC
main = {}
for sm in SUMS:
    for mn in MODELS:
        col = f"ext_{sm}_{mn}"
        g = df.dropna(subset=[col, f"conc_{sm}"]).groupby("drug")
        agg = g.agg(conc=(f"conc_{sm}", "mean"), ext=(col, "mean"),
                    internal=(f"int_{mn}", "mean"), n_split=(col, "size"))
        agg = agg[agg.n_split >= max(3, out["n_splits"] // 2)]
        rho, p = stats.spearmanr(agg.conc, agg.ext)
        per_split = [stats.spearmanr(x[f"conc_{sm}"], x[col]).statistic
                     for _, x in df.dropna(subset=[col]).groupby("split") if len(x) >= 20]
        main[f"{sm}|{mn}"] = dict(
            n_drugs=len(agg), spearman=round(float(rho), 4), p=float(p), ci=boot_spearman(agg.conc, agg.ext),
            per_split_median=round(float(np.median(per_split)), 4),
            per_split_range=[round(float(min(per_split)), 4), round(float(max(per_split)), 4)],
            partial_internal=partial_rank(agg.conc, agg.ext, [agg.internal]),
            median_ext_auroc=round(float(agg.ext.median()), 3))
out["concordance_vs_external"] = main

# ---------------- leaky vs independent concordance (fitted, RF)
col = "ext_fitted_Random Forest"
agg = df.dropna(subset=[col]).groupby("drug").agg(
    ind=("conc_fitted", "mean"), leaky=("conc_fitted_leaky_allshared", "mean"), ext=(col, "mean"))
out["leaky_vs_independent_RF_fitted"] = dict(
    spearman_independent=round(float(stats.spearmanr(agg.ind, agg.ext).statistic), 4),
    spearman_leaky=round(float(stats.spearmanr(agg.leaky, agg.ext).statistic), 4),
    median_abs_diff_concordance=round(float((agg.ind - agg.leaky).abs().median()), 4))

# ---------------- reference-size sweep (fitted, RF): concordance from k reference lines
sweep = {}
for k in (20, 50, 100):
    c = f"conc_fitted_ref{k}"
    a = df.dropna(subset=[col, c]).groupby("drug").agg(conc=(c, "mean"), sd=(f"{c}_sd", "mean"), ext=(col, "mean"))
    # per-split single-draw analogue is approximated by the mean draw SD reported alongside
    sweep[f"ref{k}"] = dict(n_drugs=len(a), spearman=round(float(stats.spearmanr(a.conc, a.ext).statistic), 4),
                            ci=boot_spearman(a.conc, a.ext),
                            median_within_drug_sd_of_concordance=round(float(a.sd.median()), 4))
a = df.dropna(subset=[col]).groupby("drug").agg(conc=("conc_fitted", "mean"), n=("n_ref_fitted", "median"), ext=(col, "mean"))
sweep["full_reference"] = dict(median_n_ref=float(a.n.median()),
                               spearman=round(float(stats.spearmanr(a.conc, a.ext).statistic), 4))
out["reference_size_sweep_RF_fitted"] = sweep

# ---------------- screening: concordance >= 0.5 -> external AUROC >= 0.70 (per split, drug level)
scr = []
for s, x in df.dropna(subset=[col]).groupby("split"):
    y = (x[col] >= THR_AUROC).astype(int); pred = (x.conc_fitted >= THR_CONC).astype(int)
    auc = roc_auc_score(y, x.conc_fitted) if y.nunique() == 2 else np.nan
    # nested threshold: pick Youden-optimal threshold on half of drugs, apply to other half
    idx = rng.permutation(len(x)); h1, h2 = idx[: len(x) // 2], idx[len(x) // 2:]
    cands = np.unique(x.conc_fitted.to_numpy()[h1])
    yj = [((x.conc_fitted.to_numpy()[h1] >= t) & (y.to_numpy()[h1] == 1)).sum() / max(1, y.to_numpy()[h1].sum())
          + ((x.conc_fitted.to_numpy()[h1] < t) & (y.to_numpy()[h1] == 0)).sum() / max(1, (1 - y.to_numpy()[h1]).sum())
          for t in cands]
    t_star = cands[int(np.argmax(yj))]
    acc_h2 = float(((x.conc_fitted.to_numpy()[h2] >= t_star) == (y.to_numpy()[h2] == 1)).mean())
    scr.append(dict(split=s, auc=auc, tp=int(((pred == 1) & (y == 1)).sum()), fp=int(((pred == 1) & (y == 0)).sum()),
                    fn=int(((pred == 0) & (y == 1)).sum()), tn=int(((pred == 0) & (y == 0)).sum()),
                    nested_threshold=float(t_star), nested_heldout_accuracy=acc_h2))
scr = pd.DataFrame(scr); scr.to_csv(D / "screen_per_split.csv", index=False)
TP, FP, FN, TN = scr.tp.sum(), scr.fp.sum(), scr.fn.sum(), scr.tn.sum()
out["screen_RF_fitted"] = dict(
    screen_auc_median=round(float(scr.auc.median()), 3), screen_auc_range=[round(float(scr.auc.min()), 3), round(float(scr.auc.max()), 3)],
    fixed_0p5_pooled_over_splits=dict(sensitivity=round(TP / (TP + FN), 3), specificity=round(TN / (TN + FP), 3),
                                      ppv=round(TP / (TP + FP), 3), note="pooled drug x split; splits are not independent"),
    nested_threshold_median=round(float(scr.nested_threshold.median()), 3),
    nested_heldout_accuracy_median=round(float(scr.nested_heldout_accuracy.median()), 3))

# ---------------- calibration table by concordance bin (drug-level means), with Wilson CI
agg = df.dropna(subset=[col]).groupby("drug").agg(conc=("conc_fitted", "mean"), ext=(col, "mean"))
bins = pd.cut(agg.conc, [-1, 0.2, 0.35, 0.5, 0.65, 1.0])
cal = agg.assign(bin=bins, ok=(agg.ext >= THR_AUROC)).groupby("bin", observed=True).ok.agg(["sum", "size"])
out["transfer_rate_by_bin_RF_fitted"] = {str(b): dict(k=int(r["sum"]), n=int(r["size"]),
                                                       rate=round(r["sum"] / r["size"], 3), wilson95=wilson(r["sum"], r["size"]))
                                         for b, r in cal.iterrows()}

# ---------------- does the internal-external gap persist under fitted AUC?
gap = {}
for sm in SUMS:
    for mn in MODELS:
        c = f"ext_{sm}_{mn}"
        a = df.dropna(subset=[c]).groupby("drug").agg(i=(f"int_{mn}", "mean"), e=(c, "mean"))
        gap[f"{sm}|{mn}"] = dict(median_internal=round(float(a.i.median()), 3), median_external=round(float(a.e.median()), 3),
                                 median_drop_pts=round(float((a.i - a.e).median() * 100), 2))
out["internal_vs_external"] = gap

agr = pd.read_csv(D / "summary_agreement.csv")
out["gdsc2_summary_agreement"] = dict(median_fitted_vs_trapz=round(float(agr.fitted_vs_trapz.median()), 3),
                                      median_fitted_vs_legacy=round(float(agr.fitted_vs_legacy.median()), 3),
                                      min_fitted_vs_legacy=round(float(agr.fitted_vs_legacy.min()), 3),
                                      worst_legacy=agr.nsmallest(5, "fitted_vs_legacy")[["drug", "fitted_vs_legacy"]].round(3).values.tolist())

(D / "summary.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=2, default=float))
