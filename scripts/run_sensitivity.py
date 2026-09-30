#!/usr/bin/env python3
"""
Sensitivity of concordance -> transferability to analysis choices:
  * label binarization (median split / tertile-extreme / quartile-extreme)
  * number of selected features (50 / 100 / 200 / 500)
  * feature-selection method (point-biserial / mutual information / L1-Lasso)
For each setting we recompute per-drug external AUROC (random forest, leakage-free)
on the CTRP->GDSC2 panel and report Spearman(concordance, external AUROC).
Concordance is label-continuous and fixed across settings. Seed 42.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="4"
import re,json,time,warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
SEED=42; np.random.seed(SEED)
from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
CC=str(CCLE_DIR)
EXPR=f"{CC}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
CTRP=f"{CC}/Drug_sensitivity_AUC_(CTD^2)_subsetted.csv"
GDSC=f"{CC}/Drug_sensitivity_replicate-level_dose_(Sanger_GDSC2)_subsetted.csv"
OUT=Path(str(RESULTS_DIR))
R=str(OUT)
VAR_Q=0.50; MIN_LAB=150; MIN_CLASS=15
log=[]; P=lambda m="":(print(m,flush=True),log.append(str(m)))
def bn(c,p): return re.split(rf"\s*\({p}:",c)[0].strip().upper()

def binarize(s, mode):
    s=s.dropna()
    if mode=="median":
        m=s.median(); y=(s>m).astype(int); return y            # all cells, high=resistant
    q=(1/3,2/3) if mode=="tertile" else (0.25,0.75)
    lo,hi=s.quantile(q[0]),s.quantile(q[1])
    idx=s[s<=lo].index.append(s[s>=hi].index); y=pd.Series(0,index=idx); y.loc[s[s>=hi].index]=1
    return y

def rf(): return RandomForestClassifier(n_estimators=300,n_jobs=4,random_state=SEED)

def select(Xtr,ytr,k,method):
    v=Xtr.var(0); keep=np.where(v>=np.quantile(v,VAR_Q))[0]; Xk=Xtr[:,keep]; yv=ytr.astype(float)
    if method=="pbiserial":
        n1=yv.sum(); m1=Xk[yv==1].mean(0); m0=Xk[yv==0].mean(0); sd=Xk.std(0); sd[sd==0]=np.nan
        r=np.nan_to_num(np.abs((m1-m0)/sd)); return keep[np.argsort(r)[::-1][:k]]
    if method=="mutual_info":
        vv=Xk.var(0); cand=np.argsort(vv)[::-1][:2000]      # cap to top-variance genes for tractable k-NN MI
        mi=mutual_info_classif(Xk[:,cand],ytr,random_state=SEED)
        return keep[cand[np.argsort(mi)[::-1][:k]]]
    if method=="lasso":
        l=LogisticRegression(penalty="l1",solver="liblinear",C=0.1,random_state=SEED).fit(
            StandardScaler().fit_transform(Xk),ytr)
        co=np.abs(l.coef_[0]); return keep[np.argsort(co)[::-1][:k]]

P(f"[{time.strftime('%H:%M:%S')}] loading expression ...")
expr=pd.read_csv(EXPR)
if "IsDefaultEntryForModel" in expr.columns:
    expr=expr[expr["IsDefaultEntryForModel"].astype(str).str.strip()=="Yes"]
META={"","SequencingID","ModelID","IsDefaultEntryForModel","ModelConditionID","IsDefaultEntryForMC"}
mc=[c for c in expr.columns if c in META or str(c).lower().startswith("unnamed")]
expr=(expr.dropna(subset=["ModelID"]).drop_duplicates("ModelID").set_index("ModelID")
        .drop(columns=[c for c in mc if c!="ModelID"],errors="ignore").astype(np.float32))
ctrp=pd.read_csv(CTRP,index_col=0); mono=[c for c in ctrp.columns if ":" not in c.split("(")[0]]; ctrp=ctrp[mono]
c2={}
for c in ctrp.columns:
    nm=bn(c,"CTRP")
    if nm not in c2 or ctrp[c].notna().sum()>ctrp[c2[nm]].notna().sum(): c2[nm]=c
gd=pd.read_csv(GDSC,index_col=0); gd.index=gd.index.astype(str); gg={}
for c in gd.columns: gg.setdefault(bn(c,"GDSC2"),[]).append(c)
gdm=pd.DataFrame({k:gd[v].mean(1,skipna=True) for k,v in gg.items()},index=gd.index)
shared=sorted(set(c2)&set(gdm.columns))
conc={r["drug"]:float(r["spearman_ctrp_gdsc2"]) for _,r in
      pd.read_csv(f"{R}/per_drug_external.csv").iterrows()
      if pd.notna(r["spearman_ctrp_gdsc2"])}

def run_setting(bin_mode,k,method):
    xr=[]; xe=[]
    for nm in shared:
        yc=binarize(ctrp[c2[nm]],bin_mode); yc=yc[yc.index.isin(expr.index)]
        yg=binarize(gdm[nm],bin_mode); yg=yg[yg.index.isin(expr.index)]
        if len(yc)<MIN_LAB or yc.nunique()<2 or min(np.bincount(yc.to_numpy()))<25: continue
        X=expr.loc[yc.index].to_numpy(); yv=yc.to_numpy()
        Xtr,Xte,ytr,yte,itr,ite=train_test_split(X,yv,yc.index.to_numpy(),test_size=0.3,stratify=yv,random_state=SEED)
        feat=select(Xtr,ytr,k,method); sc=StandardScaler().fit(Xtr[:,feat])
        eidx=yg.index[~yg.index.isin(set(itr))]; ye=yg.loc[eidx]
        if ye.nunique()<2 or min(np.bincount(ye.to_numpy()))<MIN_CLASS: continue
        c=rf().fit(sc.transform(Xtr[:,feat]),ytr)
        ext=roc_auc_score(ye.to_numpy(),c.predict_proba(sc.transform(expr.loc[eidx].to_numpy()[:,feat]))[:,1])
        if nm in conc: xr.append(conc[nm]); xe.append(ext)
    if len(xr)<10: return None
    sp=stats.spearmanr(xr,xe)
    return dict(n=len(xr),spearman=round(float(sp.statistic),3),p=float(sp.pvalue),
                median_ext=round(float(np.median(xe)),3))

settings=[("binarization","tertile",dict(bin_mode="tertile",k=200,method="pbiserial")),
          ("binarization","median",dict(bin_mode="median",k=200,method="pbiserial")),
          ("binarization","quartile",dict(bin_mode="quartile",k=200,method="pbiserial")),
          ("n_features","50",dict(bin_mode="tertile",k=50,method="pbiserial")),
          ("n_features","100",dict(bin_mode="tertile",k=100,method="pbiserial")),
          ("n_features","200",dict(bin_mode="tertile",k=200,method="pbiserial")),
          ("n_features","500",dict(bin_mode="tertile",k=500,method="pbiserial")),
          ("selection","point-biserial",dict(bin_mode="tertile",k=200,method="pbiserial")),
          ("selection","mutual-info",dict(bin_mode="tertile",k=200,method="mutual_info")),
          ("selection","lasso",dict(bin_mode="tertile",k=200,method="lasso"))]
res=[]
for cat,val,kw in settings:
    P(f"[{time.strftime('%H:%M:%S')}] setting {cat}={val} ...")
    r=run_setting(**kw)
    if r: r.update(category=cat,setting=val); res.append(r); P(f"    Spearman={r['spearman']} (p={r['p']:.1e}) n={r['n']} median_ext={r['median_ext']}")
pd.DataFrame(res).to_csv(OUT/"V_sensitivity.csv",index=False)
(OUT/"V_sensitivity_log.txt").write_text("\n".join(log))
P("[done]")
