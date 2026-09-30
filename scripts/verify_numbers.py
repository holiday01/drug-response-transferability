#!/usr/bin/env python3
"""Tier-1 verification: every headline number in main.tex must match source files."""
import json, pandas as pd, numpy as np
from scipy import stats
from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
R = str(RESULTS_DIR)

ext = pd.read_csv(f"{R}/per_drug_external.csv"); ext = ext[ext["n_ext"]>0].copy()
ext["rho"] = ext["spearman_ctrp_gdsc2"].astype(float)
r1 = pd.read_csv(f"{R}/R1_concordance_regression.csv")
tert = pd.read_csv(f"{R}/R2_concordance_tertiles.csv")
scr = pd.read_csv(f"{R}/R2_screening_operating_points.csv")
det = pd.read_csv(f"{R}/R5_determinants_per_drug.csv")
cons = json.load(open(f"{R}/R3_consensus_summary.json"))

checks = []
def chk(name, claim, actual, tol=0.02):
    ok = abs(claim-actual) <= tol
    checks.append((ok, name, claim, round(actual,4)))

# R1
rf = r1[r1.model=="Random Forest"].iloc[0]
chk("R1 RF Spearman=0.81", 0.81, rf.spearman)
chk("R1 min Spearman (MLP)=0.79", 0.79, r1[r1.model=="DNN (MLP)"].iloc[0].spearman)
chk("R1 max Spearman (XGB)=0.84", 0.84, r1[r1.model=="XGBoost"].iloc[0].spearman)
chk("R1 OLS R2 min=0.62", 0.62, r1[r1.model.isin(['Logistic Regression','Random Forest','XGBoost','SVM (RBF)','DNN (MLP)'])].ols_r2.min())
chk("R1 OLS R2 max=0.69", 0.69, r1[r1.model.isin(['Logistic Regression','Random Forest','XGBoost','SVM (RBF)','DNN (MLP)'])].ols_r2.max())
# partial from meta
meta = json.load(open(f"{R}/transferability_analysis_meta.json"))
chk("R1 partial Spearman=0.58", 0.58, meta["R1_partial_spearman"])

# R2 tertiles
lo = tert[tert.tertile=="low"].iloc[0]; hi = tert[tert.tertile=="high"].iloc[0]
chk("R2 low drop=7.6", 7.6, lo.median_drop_pts, tol=0.15)
chk("R2 high drop=3.2", 3.2, hi.median_drop_pts, tol=0.15)
chk("R2 high external=0.788", 0.788, hi.median_external)
chk("R2 low rho~0.22", 0.22, lo.rho_median, tol=0.02)
chk("R2 high rho~0.61", 0.61, hi.rho_median, tol=0.02)
# R2 screen
rf5 = scr[(scr.scored_model=="Random Forest")&(np.isclose(scr.threshold,0.5))].iloc[0]
chk("R2 screen ROC-AUC=0.91", 0.91, rf5.roc_auc_of_screen)
chk("R2 deploy n=21", 21, rf5.n_selected, tol=0)
chk("R2 deploy precision=0.905", 0.905, rf5.precision)
chk("R2 deploy external=0.784", 0.784, rf5.subset_median_external)
chk("R2 deploy drop=3.2", 3.2, rf5.subset_median_drop_pts, tol=0.15)
rf4 = scr[(scr.scored_model=="Random Forest")&(np.isclose(scr.threshold,0.4))].iloc[0]
chk("R2 thr0.4 n=34", 34, rf4.n_selected, tol=0)
chk("R2 thr0.4 recall=0.897", 0.897, rf4.recall)
chk("R2 thr0.4 precision=0.765", 0.765, rf4.precision)
chk("R2 thr0.4 external=0.749", 0.749, rf4.subset_median_external)
chk("R2 full panel external=0.696", 0.696, float(np.median(ext["ext_Random Forest"])))
chk("R2 discarded external=0.626", 0.626, meta["R2_discarded_external_RF"])
# model-avg screen 0.92
extm = scr[(scr.scored_model=="ext_mean")].iloc[0]
chk("R2 model-avg screen ROC-AUC=0.92", 0.92, extm.roc_auc_of_screen)

# R5 determinants
def sp(a):
    s=det[["rho",a]].dropna(); return stats.spearmanr(s["rho"],s[a]).statistic
chk("R5 GDSC dynrange Spearman=0.79", 0.79, sp("gdsc_dynrange"))
chk("R5 CTRP dynrange Spearman=0.57", 0.57, sp("ctrp_dynrange"))
chk("R5 lineage entropy Spearman=-0.33", -0.33, sp("lineage_entropy"))

# R3 consensus
lr = cons["Logistic Regression|cons"]; rff = cons["Random Forest|cons"]
chk("R3 LR single=0.775", 0.775, lr["median_S"])
chk("R3 LR consensus=0.853", 0.853, lr["median_C"])
chk("R3 LR delta=+5.9", 5.9, lr["delta_C_minus_S_pts"], tol=0.15)
chk("R3 LR gdsc delta=+3.6", 3.6, cons["Logistic Regression|gdsc2"]["delta_C_minus_S_pts"], tol=0.15)
chk("R3 RF single=0.849", 0.849, rff["median_S"])
chk("R3 RF consensus=0.864", 0.864, rff["median_C"])
chk("R3 median concordance=0.46", 0.46, float(ext["rho"].median()), tol=0.01)

print(f"{'OK':>4}  {'CHECK':<40}{'CLAIM':>10}{'ACTUAL':>10}")
nfail=0
for ok,name,claim,actual in checks:
    print(f"{'✓' if ok else '✗ FAIL':>4}  {name:<40}{claim:>10}{actual:>10}")
    if not ok: nfail+=1
print(f"\n{len(checks)-nfail}/{len(checks)} passed; {nfail} FAILED")
