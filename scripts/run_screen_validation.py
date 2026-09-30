#!/usr/bin/env python3
"""
Prospective evaluation of the concordance screen (no expression needed):
leave-one-drug-out (LODO) shows concordance predicts transferability for drugs
never used to set the rule, plus PR-AUC and calibration.
"""
import json, numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
R=str(RESULTS_DIR)
ext=pd.read_csv(f"{R}/per_drug_external.csv"); ext=ext[ext.n_ext>0].copy()
ext["rho"]=ext["spearman_ctrp_gdsc2"].astype(float)
ext["ext_mean"]=ext[[f"ext_{m}" for m in
    ["Logistic Regression","Random Forest","XGBoost","SVM (RBF)","DNN (MLP)"]]].mean(1)
THR=0.70; PRIMARY="Random Forest"
y_transfer=(ext[f"ext_{PRIMARY}"]>=THR).astype(int).values
rho=ext["rho"].values.reshape(-1,1)
ext_cont=ext[f"ext_{PRIMARY}"].values
log=[]
def P(m=""): print(m,flush=True); log.append(str(m))

# ---- LODO classification: predict each drug's transferability from the others
pred=np.zeros(len(ext))
for i in range(len(ext)):
    tr=np.ones(len(ext),bool); tr[i]=False
    if len(set(y_transfer[tr]))<2:
        pred[i]=y_transfer[tr].mean(); continue
    clf=LogisticRegression().fit(rho[tr], y_transfer[tr])
    pred[i]=clf.predict_proba(rho[i:i+1])[0,1]
lodo_auc=roc_auc_score(y_transfer, pred)
lodo_pr=average_precision_score(y_transfer, pred)
insample_auc=roc_auc_score(y_transfer, ext["rho"].values)
P(f"[LODO classification] transferable = external {PRIMARY} >= {THR}")
P(f"  in-sample concordance ROC-AUC        = {insample_auc:.3f}")
P(f"  leave-one-drug-out ROC-AUC           = {lodo_auc:.3f}")
P(f"  leave-one-drug-out PR-AUC (base {y_transfer.mean():.2f}) = {lodo_pr:.3f}")

# ---- LODO regression: predict continuous external AUROC from concordance
pred_c=np.zeros(len(ext))
for i in range(len(ext)):
    tr=np.ones(len(ext),bool); tr[i]=False
    reg=LinearRegression().fit(rho[tr], ext_cont[tr])
    pred_c[i]=reg.predict(rho[i:i+1])[0]
ss_res=((ext_cont-pred_c)**2).sum(); ss_tot=((ext_cont-ext_cont.mean())**2).sum()
lodo_r2=1-ss_res/ss_tot; rmse=np.sqrt(((ext_cont-pred_c)**2).mean())
P(f"\n[LODO regression] predict continuous external AUROC from concordance")
P(f"  out-of-sample R2  = {lodo_r2:.3f}")
P(f"  out-of-sample RMSE= {rmse:.3f} AUROC points = {rmse*100:.1f}")

# ---- calibration by concordance bins
ext["bin"]=pd.cut(ext["rho"], [-1,0.3,0.45,0.6,1.0], labels=["<0.30","0.30-0.45","0.45-0.60",">0.60"])
cal=ext.groupby("bin",observed=True).agg(
    n=("drug","size"), mean_rho=("rho","mean"),
    transfer_rate=(f"ext_{PRIMARY}", lambda s:(s>=THR).mean()),
    median_external=(f"ext_{PRIMARY}","median")).reset_index()
cal.to_csv(f"{R}/R2b_screen_calibration.csv", index=False)
P(f"\n[calibration] observed transfer rate by concordance bin:")
for _,r in cal.iterrows():
    P(f"  rho {str(r['bin']):11s} n={int(r.n):2d}  transfer_rate={r.transfer_rate:.2f}  "
      f"median_external={r.median_external:.3f}")

summ=dict(threshold=THR, insample_roc_auc=round(float(insample_auc),3),
          lodo_roc_auc=round(float(lodo_auc),3), lodo_pr_auc=round(float(lodo_pr),3),
          base_rate=round(float(y_transfer.mean()),3),
          lodo_regression_r2=round(float(lodo_r2),3),
          lodo_regression_rmse_pts=round(float(rmse*100),1), n_drugs=int(len(ext)))
json.dump(summ, open(f"{R}/R2b_screen_validation.json","w"), indent=2)
open(f"{R}/R2b_screen_validation_log.txt","w").write("\n".join(log))
P(f"\n[done]")
