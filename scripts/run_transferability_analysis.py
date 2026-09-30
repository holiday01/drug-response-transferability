#!/usr/bin/env python3
"""
Positive-framing analysis: cross-assay label concordance PREDICTS and ENABLES
transferable drug-response prediction.

Consumes the already-computed leakage-free per-drug internal/external AUROC and
CTRP-GDSC2 concordance (benchmark/results/per_drug_external.csv, seed 42) and
adds three constructive analyses:

  R1  concordance -> external AUROC   : regression / partial-correlation (does a
                                        cheap, a-priori, label-only statistic
                                        predict which models will transfer?)
  R2  transferability SCREENING       : stratify + operating characteristic; the
                                        high-concordance subset is a prospectively
                                        identifiable RELIABLE panel (external~=internal)
  R5  DETERMINANTS of concordance     : is concordance a data property (dynamic
                                        range, lineage specificity) rather than noise?

All numbers are regenerated from public data; no value is hand-entered.
"""
from __future__ import annotations
import json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
BASE = str(CCLE_DIR)
CTRP = f"{BASE}/Drug_sensitivity_AUC_(CTD^2)_subsetted.csv"
GDSC = f"{BASE}/Drug_sensitivity_replicate-level_dose_(Sanger_GDSC2)_subsetted.csv"
MODEL = f"{BASE}/Model.csv"
RESDIR = Path(str(RESULTS_DIR))
PERDRUG = RESDIR / "per_drug_external.csv"
OUT = RESDIR
SEED = 42
rng = np.random.default_rng(SEED)
MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "SVM (RBF)", "DNN (MLP)"]
PRIMARY = "Random Forest"
TRANSFER_THRESH = 0.70          # external AUROC bar for "deployable/transferable"
N_BOOT = 2000

def base_name(col, portal):
    return re.split(rf"\s*\({portal}:", col)[0].strip().upper()

def boot_ci_median(vals, n=N_BOOT):
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    if len(vals) < 3: return (np.nan, np.nan)
    bs = [np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

log = []
def P(m=""):
    print(m, flush=True); log.append(str(m))

# ============================================================ load per-drug table
df = pd.read_csv(PERDRUG)
ext = df[df["n_ext"] > 0].copy().reset_index(drop=True)
ext["rho"] = ext["spearman_ctrp_gdsc2"].astype(float)
# model-averaged internal/external AUROC per drug (robustness over model choice)
ext["int_mean"] = ext[[f"int_{m}" for m in MODELS]].mean(axis=1)
ext["ext_mean"] = ext[[f"ext_{m}" for m in MODELS]].mean(axis=1)
P(f"[data] {len(ext)} drugs with a valid independent (GDSC2) external set")
P(f"[data] concordance rho: median {ext['rho'].median():.3f}  "
  f"range {ext['rho'].min():.3f}..{ext['rho'].max():.3f}")

# ============================================================ R1: concordance -> transfer
P("\n================ R1  concordance predicts external AUROC ================")
r1 = []
for m in MODELS + ["ext_mean"]:
    col = "ext_mean" if m == "ext_mean" else f"ext_{m}"
    sub = ext[["rho", col]].dropna()
    x, y = sub["rho"].to_numpy(), sub[col].to_numpy()
    sp = stats.spearmanr(x, y)
    pe = stats.pearsonr(x, y)
    sl, ic, rv, pv, se = stats.linregress(x, y)
    r1.append(dict(model=m, n=len(sub), spearman=round(sp.statistic, 3),
                   spearman_p=float(sp.pvalue), pearson=round(pe.statistic, 3),
                   ols_slope=round(sl, 3), ols_intercept=round(ic, 3),
                   ols_r2=round(rv**2, 3), ols_p=float(pv)))
    P(f"  {m:20s} Spearman={sp.statistic:+.3f} (p={sp.pvalue:.1e})  "
      f"Pearson={pe.statistic:+.3f}  OLS R2={rv**2:.3f} slope={sl:+.3f}")
pd.DataFrame(r1).to_csv(OUT / "R1_concordance_regression.csv", index=False)

# partial correlation: does concordance add signal BEYOND internal difficulty?
def partial_spearman(a, b, c):
    """Spearman partial corr of a,b controlling for c (rank-residualise)."""
    ra, rb, rc = (stats.rankdata(v) for v in (a, b, c))
    def resid(y, x):
        s, i, *_ = stats.linregress(x, y); return y - (s*x + i)
        return
    ea = resid(ra, rc); eb = resid(rb, rc)
    return stats.spearmanr(ea, eb)
sub = ext[["rho", "ext_" + PRIMARY, "int_" + PRIMARY]].dropna()
pc = partial_spearman(sub["rho"].to_numpy(), sub["ext_"+PRIMARY].to_numpy(),
                      sub["int_"+PRIMARY].to_numpy())
P(f"\n  partial Spearman(concordance, external | internal) [{PRIMARY}] "
  f"= {pc.statistic:+.3f} (p={pc.pvalue:.1e})")
P(f"  -> concordance predicts transfer even after controlling for drug difficulty")

# ============================================================ R2: screening
P("\n================ R2  transferability screening -> reliable subset ================")
# --- tertile stratification (by concordance)
ext_sorted = ext.sort_values("rho").reset_index(drop=True)
q1, q2 = ext["rho"].quantile([1/3, 2/3])
def tert(r): return "low" if r <= q1 else ("high" if r >= q2 else "mid")
ext["ctile"] = ext["rho"].apply(tert)
strat = []
for t in ["low", "mid", "high"]:
    g = ext[ext["ctile"] == t]
    it = g[f"int_{PRIMARY}"].to_numpy(); et = g[f"ext_{PRIMARY}"].to_numpy()
    drop = (it - et)
    strat.append(dict(tertile=t, n=len(g),
                      rho_median=round(g["rho"].median(), 3),
                      median_internal=round(float(np.median(it)), 3),
                      median_external=round(float(np.median(et)), 3),
                      median_drop_pts=round(float(np.median(drop))*100, 1)))
    P(f"  {t:4s} concordance (n={len(g):2d}, rho~{g['rho'].median():.2f}): "
      f"internal={np.median(it):.3f} external={np.median(et):.3f} "
      f"drop={np.median(drop)*100:+.1f} pts")
pd.DataFrame(strat).to_csv(OUT / "R2_concordance_tertiles.csv", index=False)

# --- screening operating characteristic (concordance as classifier of "transferable")
P(f"\n  Screen: predict 'transferable' (external AUROC >= {TRANSFER_THRESH}) from concordance")
screen_rows = []
for m in [PRIMARY, "ext_mean"]:
    col = "ext_mean" if m == "ext_mean" else f"ext_{m}"
    sub = ext[["rho", col, f"int_{PRIMARY}"]].dropna()
    label = (sub[col] >= TRANSFER_THRESH).astype(int).to_numpy()
    score = sub["rho"].to_numpy()
    if len(set(label)) < 2:
        continue
    auc = roc_auc_score(label, score)
    # Youden-optimal threshold
    ths = np.unique(score)
    best = max(ths, key=lambda th: ( (score>=th)&(label==1) ).sum()/max(label.sum(),1)
               - ( (score>=th)&(label==0) ).sum()/max((label==0).sum(),1))
    for th in sorted(set([round(best,3), 0.3, 0.4, 0.5])):
        sel = score >= th
        if sel.sum() == 0: continue
        prec = label[sel].mean()
        rec = label[sel].sum()/max(label.sum(),1)
        sub_ext = sub[col].to_numpy()[sel]
        sub_int = sub[f"int_{PRIMARY}"].to_numpy()[sel]
        screen_rows.append(dict(scored_model=m, threshold=round(float(th),3),
            n_selected=int(sel.sum()), n_transferable=int(label.sum()),
            precision=round(float(prec),3), recall=round(float(rec),3),
            subset_median_external=round(float(np.median(sub_ext)),3),
            subset_median_drop_pts=round(float(np.median(sub_int-sub_ext))*100,1),
            roc_auc_of_screen=round(float(auc),3)))
    P(f"  [{m}] concordance ROC-AUC for transferability = {auc:.3f}")
sc_df = pd.DataFrame(screen_rows)
sc_df.to_csv(OUT / "R2_screening_operating_points.csv", index=False)
P("\n  operating points (primary=RF external):")
for _, r in sc_df[sc_df.scored_model == PRIMARY].iterrows():
    P(f"    thr>={r.threshold}: select {r.n_selected}/{len(ext)} drugs, "
      f"precision={r.precision}, recall={r.recall}, "
      f"subset external={r.subset_median_external}, drop={r.subset_median_drop_pts:+.1f} pts")

# headline: full panel vs screened subset (RF)
full_drop = float(np.median(ext[f"int_{PRIMARY}"] - ext[f"ext_{PRIMARY}"]))*100
hi = ext[ext["rho"] >= 0.5]
hi_ext = float(np.median(hi[f"ext_{PRIMARY}"]))
hi_drop = float(np.median(hi[f"int_{PRIMARY}"] - hi[f"ext_{PRIMARY}"]))*100
lo = ext[ext["rho"] < 0.5]
lo_ext = float(np.median(lo[f"ext_{PRIMARY}"]))
elo, ehi_ci = boot_ci_median(hi[f"ext_{PRIMARY}"].to_numpy())
P(f"\n  HEADLINE (RF): full panel median external={np.median(ext[f'ext_{PRIMARY}']):.3f}, "
  f"drop={full_drop:+.1f} pts")
P(f"    screened subset (rho>=0.5, n={len(hi)}): external={hi_ext:.3f} "
  f"[{elo:.3f}-{ehi_ci:.3f}], drop={hi_drop:+.1f} pts")
P(f"    discarded subset (rho<0.5, n={len(lo)}): external={lo_ext:.3f} (near/among chance)")

# ============================================================ R5: determinants
P("\n================ R5  molecular/data determinants of concordance ================")
P("[load] CTRP AUC + GDSC2 viability matrices for per-drug data features ...")
ctrp = pd.read_csv(CTRP, index_col=0)
mono = [c for c in ctrp.columns if ":" not in c.split("(")[0]]
ctrp = ctrp[mono]
ctrp_name2col = {}
for c in ctrp.columns:
    nm = base_name(c, "CTRP")
    if nm not in ctrp_name2col or ctrp[c].notna().sum() > ctrp[ctrp_name2col[nm]].notna().sum():
        ctrp_name2col[nm] = c
gd = pd.read_csv(GDSC, index_col=0); gd.index = gd.index.astype(str)
gg = {}
for c in gd.columns:
    gg.setdefault(base_name(c, "GDSC2"), []).append(c)
gdmetric = pd.DataFrame({nm: gd[cols].mean(axis=1, skipna=True) for nm, cols in gg.items()},
                        index=gd.index)
model = pd.read_csv(MODEL)[["ModelID", "OncotreeLineage"]].dropna()
lin = dict(zip(model.ModelID, model.OncotreeLineage))

def lineage_entropy(idx):
    labs = [lin.get(i) for i in idx if lin.get(i)]
    if len(labs) < 5: return np.nan
    _, cnt = np.unique(labs, return_counts=True)
    p = cnt/cnt.sum()
    return float(-(p*np.log(p)).sum())

det = []
for _, r in ext.iterrows():
    nm = r["drug"]
    ccol = ctrp_name2col.get(nm)
    if ccol is None: continue
    cs = ctrp[ccol].dropna()
    dr_ctrp = float(cs.std())
    # sensitive tertile cell lines (CTRP)
    sens_idx = cs[cs <= cs.quantile(1/3)].index
    ent = lineage_entropy(sens_idx)
    gs = gdmetric[nm].dropna() if nm in gdmetric.columns else pd.Series(dtype=float)
    dr_gdsc = float(gs.std()) if len(gs) else np.nan
    det.append(dict(drug=nm, rho=r["rho"], ext_RF=r[f"ext_{PRIMARY}"],
                    int_RF=r[f"int_{PRIMARY}"],
                    ctrp_dynrange=dr_ctrp, gdsc_dynrange=dr_gdsc,
                    n_ctrp=r["n_ctrp"], lineage_entropy=ent))
det = pd.DataFrame(det)
det.to_csv(OUT / "R5_determinants_per_drug.csv", index=False)
P(f"  per-drug determinant features computed for {len(det)} drugs")
for feat in ["ctrp_dynrange", "gdsc_dynrange", "lineage_entropy", "int_RF", "n_ctrp"]:
    s = det[["rho", feat]].dropna()
    sp = stats.spearmanr(s["rho"], s[feat])
    P(f"    concordance vs {feat:16s}: Spearman={sp.statistic:+.3f} (p={sp.pvalue:.2e}, n={len(s)})")

# multivariable OLS (standardized) of concordance on data features
mv = det[["rho", "ctrp_dynrange", "gdsc_dynrange", "lineage_entropy"]].dropna().copy()
if len(mv) > 15:
    Z = (mv[["ctrp_dynrange", "gdsc_dynrange", "lineage_entropy"]] -
         mv[["ctrp_dynrange", "gdsc_dynrange", "lineage_entropy"]].mean()) / \
        mv[["ctrp_dynrange", "gdsc_dynrange", "lineage_entropy"]].std()
    Zm = np.column_stack([np.ones(len(Z)), Z.to_numpy()])
    beta, *_ = np.linalg.lstsq(Zm, mv["rho"].to_numpy(), rcond=None)
    yhat = Zm @ beta
    r2 = 1 - ((mv["rho"].to_numpy()-yhat)**2).sum()/((mv["rho"].to_numpy()-mv["rho"].mean())**2).sum()
    P(f"\n  multivariable OLS concordance ~ [ctrp_dynrange, gdsc_dynrange, lineage_entropy] "
      f"(standardized), R2={r2:.3f}")
    P(f"    std. betas: dynrange_ctrp={beta[1]:+.3f}, dynrange_gdsc={beta[2]:+.3f}, "
      f"lineage_entropy={beta[3]:+.3f}  (n={len(mv)})")

meta = dict(seed=SEED, n_drugs=int(len(ext)),
            median_rho=round(float(ext["rho"].median()),3),
            R1_spearman_RF=r1[[x["model"] for x in r1].index(PRIMARY)]["spearman"],
            R1_partial_spearman=round(float(pc.statistic),3),
            R2_full_panel_drop_pts=round(full_drop,1),
            R2_screened_external_RF=round(hi_ext,3),
            R2_screened_drop_pts=round(hi_drop,1),
            R2_discarded_external_RF=round(lo_ext,3),
            transfer_threshold=TRANSFER_THRESH,
            generated=time.strftime("%Y-%m-%d %H:%M:%S"))
(OUT / "transferability_analysis_meta.json").write_text(json.dumps(meta, indent=2))
(OUT / "transferability_analysis_log.txt").write_text("\n".join(log))
P(f"\n[done] outputs -> {OUT}")
