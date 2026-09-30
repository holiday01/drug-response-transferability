#!/usr/bin/env python3
"""
REV extras: manuscript tables under the independent design (REV-1, fitted GDSC2).
  * Table 1: Spearman / Pearson / OLS per model (drug-level means over splits)
  * Table 2: screen operating points at concordance thresholds (RF)
  * leave-one-drug-out screen: logistic ROC-AUC and linear out-of-sample R2 / RMSE
  * baselines (same splits, reference lines only): internal AUROC, CTRP and GDSC2
    dynamic range (SD of the response on reference lines)
  * three-assay per-pair Spearman (REV-2)
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import load_ctrp, load_gdsc2

SEED = 42
from config import RESULTS_DIR
R = RESULTS_DIR
D1 = R / "rev1_independent"
df = pd.read_csv(D1 / "per_drug_split.csv")
ids = pd.read_csv(D1 / "split_ids.csv.gz")
MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "SVM (RBF)", "DNN (MLP)"]
out = {}

# ---------------- Table 1
t1 = {}
avg = []
for mn in MODELS:
    c = f"ext_fitted_{mn}"
    a = df.dropna(subset=[c]).groupby("drug").agg(conc=("conc_fitted", "mean"), e=(c, "mean"))
    avg.append(a.e.rename(mn))
    lr = stats.linregress(a.conc, a.e)
    t1[mn] = dict(n=len(a), spearman=round(stats.spearmanr(a.conc, a.e).statistic, 3),
                  pearson=round(stats.pearsonr(a.conc, a.e).statistic, 3),
                  ols_r2=round(lr.rvalue ** 2, 3), ols_slope=round(lr.slope, 3))
m = pd.concat(avg, axis=1).mean(axis=1)
conc = df.groupby("drug").conc_fitted.mean().loc[m.index]
lr = stats.linregress(conc, m)
t1["Model average"] = dict(n=len(m), spearman=round(stats.spearmanr(conc, m).statistic, 3),
                           pearson=round(stats.pearsonr(conc, m).statistic, 3),
                           ols_r2=round(lr.rvalue ** 2, 3), ols_slope=round(lr.slope, 3))
out["table1"] = t1

# ---------------- drug-level RF frame
c = "ext_fitted_Random Forest"
a = df.dropna(subset=[c]).groupby("drug").agg(conc=("conc_fitted", "mean"), e=(c, "mean"),
                                              i=("int_Random Forest", "mean"))
a["ok"] = (a.e >= 0.70).astype(int)
a["drop"] = a.i - a.e
a.to_csv(D1 / "drug_level_RF.csv")
out["screen_auc_drug_level"] = round(roc_auc_score(a.ok, a.conc), 3)
avg_ok = (m >= 0.70).astype(int)
out["screen_auc_model_average"] = round(roc_auc_score(avg_ok, conc), 3)
t2 = [dict(threshold="full", n=len(a), precision=None, recall=None,
           ext=round(a.e.median(), 3), drop_pts=round(a["drop"].median() * 100, 1))]
for t in (0.3, 0.4, 0.5):
    s = a[a.conc >= t]
    t2.append(dict(threshold=t, n=len(s), precision=round(s.ok.mean(), 3),
                   recall=round(s.ok.sum() / a.ok.sum(), 3), ext=round(s.e.median(), 3),
                   drop_pts=round(s["drop"].median() * 100, 1)))
out["table2"] = t2
out["n_transferable"] = int(a.ok.sum())
lo = a[a.conc < 0.5]
out["discarded_panel_median_ext"] = round(lo.e.median(), 3)
q = pd.qcut(a.conc, 3, labels=["low", "mid", "high"])
out["tertiles"] = {k: dict(median_conc=round(g.conc.median(), 2), median_int=round(g.i.median(), 3),
                           median_ext=round(g.e.median(), 3), drop_pts=round(g["drop"].median() * 100, 1))
                   for k, g in a.groupby(q, observed=True)}

# ---------------- leave-one-drug-out
p_cls, p_reg = [], []
X = a[["conc"]].to_numpy()
for j in range(len(a)):
    tr = np.arange(len(a)) != j
    p_cls.append(LogisticRegression().fit(X[tr], a.ok.values[tr]).predict_proba(X[j:j + 1])[0, 1])
    p_reg.append(LinearRegression().fit(X[tr], a.e.values[tr]).predict(X[j:j + 1])[0])
p_cls, p_reg = np.array(p_cls), np.array(p_reg)
ss = ((a.e.values - p_reg) ** 2).sum(); st = ((a.e.values - a.e.mean()) ** 2).sum()
brier = float(np.mean((p_cls - a.ok.values) ** 2))
cal = pd.DataFrame(dict(p=p_cls, y=a.ok.values)).assign(bin=lambda x: pd.cut(x.p, [0, .25, .5, .75, 1.0], include_lowest=True))
out["lodo"] = dict(roc_auc=round(roc_auc_score(a.ok, p_cls), 3), r2=round(1 - ss / st, 3),
                   rmse_pts=round(float(np.sqrt(ss / len(a))) * 100, 1), brier=round(brier, 3),
                   brier_null=round(float(a.ok.mean() * (1 - a.ok.mean())), 3),
                   calibration_bins={str(k): dict(n=int(len(g)), mean_pred=round(g.p.mean(), 3),
                                                  observed=round(g.y.mean(), 3))
                                     for k, g in cal.groupby("bin", observed=True)})

# ---------------- baselines on the same splits (reference lines only)
ctrp, c2col = load_ctrp()
resp, _, _ = load_gdsc2(sorted(df.drug.unique()), log=lambda m: None)
rows = []
for _, r in ids.iterrows():
    T = set(r.test.split(";")); nm = r.drug
    cv = ctrp[c2col[nm]].dropna(); gv = resp["fitted"][nm]
    rows.append(dict(drug=nm, split=r.split,
                     ctrp_dr=cv[~cv.index.isin(T)].std(), gdsc2_dr=gv[~gv.index.isin(T)].std()))
b = pd.DataFrame(rows).groupby("drug").mean(numeric_only=True)
a = a.join(b)
base = {}
for k in ["conc", "i", "ctrp_dr", "gdsc2_dr"]:
    pr = []
    for j in range(len(a)):
        tr = np.arange(len(a)) != j
        pr.append(LinearRegression().fit(a[[k]].values[tr], a.e.values[tr]).predict(a[[k]].values[j:j + 1])[0])
    pr = np.array(pr); r2 = 1 - ((a.e.values - pr) ** 2).sum() / st
    base[k] = dict(screen_auc=round(roc_auc_score(a.ok, a[k]), 3), lodo_r2=round(r2, 3),
                   spearman_with_ext=round(stats.spearmanr(a[k], a.e).statistic, 3))
# paired bootstrap of screen-AUC difference, concordance vs target dynamic range
rng = np.random.default_rng(SEED); dif = []
for _ in range(5000):
    i = rng.integers(0, len(a), len(a)); s = a.iloc[i]
    if s.ok.nunique() == 2:
        dif.append(roc_auc_score(s.ok, s.conc) - roc_auc_score(s.ok, s.gdsc2_dr))
base["conc_minus_gdsc2_dr_screen_auc_ci"] = [round(float(np.percentile(dif, 2.5)), 3), round(float(np.percentile(dif, 97.5)), 3)]
# does concordance add to target dynamic range? LODO R2 of the two-variable model
pr = []
for j in range(len(a)):
    tr = np.arange(len(a)) != j
    pr.append(LinearRegression().fit(a[["conc", "gdsc2_dr"]].values[tr], a.e.values[tr]).predict(a[["conc", "gdsc2_dr"]].values[j:j + 1])[0])
base["conc_plus_gdsc2_dr_lodo_r2"] = round(1 - ((a.e.values - np.array(pr)) ** 2).sum() / st, 3)
base["spearman_conc_vs_gdsc2_dr"] = round(stats.spearmanr(a.conc, a.gdsc2_dr).statistic, 3)
base["spearman_conc_vs_ctrp_dr"] = round(stats.spearmanr(a.conc, a.ctrp_dr).statistic, 3)
out["baselines"] = base

# ---------------- three-assay per pair (REV-2)
pp = pd.read_csv(R / "rev2_three_assay" / "per_pair_split.csv")
g = pp.groupby(["drug", "src", "dst"]).agg(rho=("rho", "first"), e=("ext_Random Forest", "mean")).reset_index()
g["pair"] = g.src + "->" + g.dst
out["three_assay_pairs"] = {p: dict(n=len(x), spearman=round(stats.spearmanr(x.rho, x.e).statistic, 3))
                            for p, x in g.groupby("pair")}
out["three_assay_pooled"] = dict(n=len(g), n_unique_drugs=int(g.drug.nunique()),
                                 spearman=round(stats.spearmanr(g.rho, g.e).statistic, 3))
(D1 / "extra_summary.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=2, default=float))
