#!/usr/bin/env python3
"""
Robustness of the concordance->transferability relationship:
  (1) BIDIRECTIONAL: train on GDSC2, test on CTRP (reverse of the main direction);
  (2) REGRESSION: continuous (non-binarised) formulation, cross-dataset Spearman.
Both ask whether per-drug CTRP-GDSC2 concordance still predicts cross-dataset
transfer. Leakage-free, seed 42; every value regenerated from public data.
"""
from __future__ import annotations
import json, re, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
SEED=42; np.random.seed(SEED)
from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
BASE=str(CCLE_DIR)
EXPR=f"{BASE}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
CTRP=f"{BASE}/Drug_sensitivity_AUC_(CTD^2)_subsetted.csv"
GDSC=f"{BASE}/Drug_sensitivity_replicate-level_dose_(Sanger_GDSC2)_subsetted.csv"
OUT=Path(str(RESULTS_DIR))
TOP_K,VAR_Q=200,0.50; MIN_LAB=150; MIN_CLASS=15
log=[]
def P(m=""): print(m,flush=True); log.append(str(m))
def base_name(col,portal): return re.split(rf"\s*\({portal}:",col)[0].strip().upper()
def tertile_binary(s):
    s=s.dropna(); lo,hi=s.quantile(1/3),s.quantile(2/3)
    idx=s[s<=lo].index.append(s[s>=hi].index); y=pd.Series(0,index=idx); y.loc[s[s>=hi].index]=1
    return y
def sel_bin(Xtr,ytr,k=TOP_K,vq=VAR_Q):
    v=Xtr.var(0); keep=np.where(v>=np.quantile(v,vq))[0]; Xk=Xtr[:,keep]; y=ytr.astype(float)
    n1=y.sum()
    if n1<2 or len(y)-n1<2: return keep[:k]
    m1=Xk[y==1].mean(0); m0=Xk[y==0].mean(0); sd=Xk.std(0); sd[sd==0]=np.nan
    r=np.nan_to_num(np.abs((m1-m0)/sd)); return keep[np.argsort(r)[::-1][:k]]
def sel_cont(Xtr,ytr,k=TOP_K,vq=VAR_Q):
    v=Xtr.var(0); keep=np.where(v>=np.quantile(v,vq))[0]; Xk=Xtr[:,keep]
    yc=ytr-ytr.mean(); Xc=Xk-Xk.mean(0)
    num=(Xc*yc[:,None]).sum(0); den=np.sqrt((Xc**2).sum(0)*(yc**2).sum())+1e-9
    r=np.abs(num/den); return keep[np.argsort(r)[::-1][:k]]

P(f"[{time.strftime('%H:%M:%S')}] loading expression ...")
expr=pd.read_csv(EXPR)
if "IsDefaultEntryForModel" in expr.columns:
    expr=expr[expr["IsDefaultEntryForModel"].astype(str).str.strip()=="Yes"]
META={"","SequencingID","ModelID","IsDefaultEntryForModel","ModelConditionID","IsDefaultEntryForMC"}
mc=[c for c in expr.columns if c in META or str(c).lower().startswith("unnamed")]
expr=(expr.dropna(subset=["ModelID"]).drop_duplicates("ModelID").set_index("ModelID")
        .drop(columns=[c for c in mc if c!="ModelID"],errors="ignore").astype(np.float32))
ctrp=pd.read_csv(CTRP,index_col=0); mono=[c for c in ctrp.columns if ":" not in c.split("(")[0]]; ctrp=ctrp[mono]
c2col={}
for c in ctrp.columns:
    nm=base_name(c,"CTRP")
    if nm not in c2col or ctrp[c].notna().sum()>ctrp[c2col[nm]].notna().sum(): c2col[nm]=c
gd=pd.read_csv(GDSC,index_col=0); gd.index=gd.index.astype(str); gg={}
for c in gd.columns: gg.setdefault(base_name(c,"GDSC2"),[]).append(c)
gdm=pd.DataFrame({nm:gd[cols].mean(1,skipna=True) for nm,cols in gg.items()},index=gd.index)
shared=sorted(set(c2col)&set(gdm.columns))
P(f"  shared drugs {len(shared)}")

# ============================================================ (1) BIDIRECTIONAL GDSC2->CTRP
P(f"\n[{time.strftime('%H:%M:%S')}] === bidirectional: train GDSC2, test CTRP ===")
MODELS={"Logistic Regression":lambda:LogisticRegression(max_iter=2000,C=1.0,random_state=SEED),
        "Random Forest":lambda:RandomForestClassifier(n_estimators=400,n_jobs=-1,random_state=SEED),
        "SVM (RBF)":lambda:SVC(kernel="rbf",C=1.0,probability=True,random_state=SEED)}
def prob(clf,X): return clf.predict_proba(X)[:,1]
rows=[]
for nm in shared:
    yg=tertile_binary(gdm[nm]); yg=yg[yg.index.isin(expr.index)]
    yc_full=tertile_binary(ctrp[c2col[nm]]); yc_full=yc_full[yc_full.index.isin(expr.index)]
    if len(yg)<MIN_LAB or yg.nunique()<2: continue
    Xg=expr.loc[yg.index].to_numpy(); yv=yg.to_numpy()
    if min(np.bincount(yv))<20: continue
    Xtr,Xte,ytr,yte,itr,ite=train_test_split(Xg,yv,yg.index.to_numpy(),test_size=0.30,stratify=yv,random_state=SEED)
    feat=sel_bin(Xtr,ytr); sc=StandardScaler().fit(Xtr[:,feat])
    Xtr_s=sc.transform(Xtr[:,feat]); Xte_s=sc.transform(Xte[:,feat])
    ext_idx=yc_full.index[~yc_full.index.isin(set(itr))]; yext=yc_full.loc[ext_idx]
    if yext.nunique()<2 or min(np.bincount(yext.to_numpy()))<MIN_CLASS: continue
    Xext_s=sc.transform(expr.loc[ext_idx].to_numpy()[:,feat])
    both=ctrp[c2col[nm]].dropna().index.intersection(gdm[nm].dropna().index)
    rho=stats.spearmanr(ctrp[c2col[nm]].loc[both],gdm[nm].loc[both]).statistic if len(both)>=20 else np.nan
    row=dict(drug=nm,rho=round(float(rho),3) if not np.isnan(rho) else None)
    for mn,mk in MODELS.items():
        clf=mk().fit(Xtr_s,ytr)
        row[f"int_{mn}"]=round(float(roc_auc_score(yte,prob(clf,Xte_s))),4)
        row[f"ext_{mn}"]=round(float(roc_auc_score(yext.to_numpy(),prob(clf,Xext_s))),4)
    rows.append(row)
bd=pd.DataFrame(rows); bd.to_csv(OUT/"R6_bidirectional_per_drug.csv",index=False)
P(f"  drugs with reverse external set: {len(bd)}")
bsum={}
for mn in MODELS:
    s=bd[["rho",f"ext_{mn}"]].dropna(); sp=stats.spearmanr(s["rho"],s[f"ext_{mn}"])
    drop=(bd[f"int_{mn}"]-bd[f"ext_{mn}"]).median()*100
    bsum[mn]=dict(spearman_rho_ext=round(float(sp.statistic),3),spearman_p=float(sp.pvalue),
                  median_internal=round(float(bd[f"int_{mn}"].median()),3),
                  median_external=round(float(bd[f"ext_{mn}"].median()),3),median_drop_pts=round(float(drop),1))
    P(f"  {mn:20s} Spearman(rho,ext_CTRP)={sp.statistic:+.3f} (p={sp.pvalue:.1e})  "
      f"int={bd[f'int_{mn}'].median():.3f} ext={bd[f'ext_{mn}'].median():.3f} drop={drop:+.1f}")

# ============================================================ (2) REGRESSION (continuous)
P(f"\n[{time.strftime('%H:%M:%S')}] === regression: continuous CTRP->GDSC2 transfer ===")
rows2=[]
for nm in shared:
    yc=ctrp[c2col[nm]].dropna(); yc=yc[yc.index.isin(expr.index)]
    if len(yc)<MIN_LAB: continue
    yg=gdm[nm].dropna(); yg=yg[yg.index.isin(expr.index)]
    X=expr.loc[yc.index].to_numpy(); yv=yc.to_numpy()
    Xtr,Xte,ytr,yte,itr,ite=train_test_split(X,yv,yc.index.to_numpy(),test_size=0.30,random_state=SEED)
    feat=sel_cont(Xtr,ytr); sc=StandardScaler().fit(Xtr[:,feat])
    reg=RandomForestRegressor(n_estimators=300,n_jobs=-1,random_state=SEED).fit(sc.transform(Xtr[:,feat]),ytr)
    # internal: predicted vs actual CTRP on held-out
    pin=reg.predict(sc.transform(Xte[:,feat])); rin=stats.spearmanr(pin,yte).statistic
    # external: predicted CTRP-sensitivity vs actual GDSC2, on GDSC2 cells not in train
    ext_idx=[i for i in yg.index if i not in set(itr)]
    if len(ext_idx)<30: continue
    pext=reg.predict(sc.transform(expr.loc[ext_idx].to_numpy()[:,feat]))
    rext=stats.spearmanr(pext,yg.loc[ext_idx]).statistic
    both=ctrp[c2col[nm]].dropna().index.intersection(gdm[nm].dropna().index)
    rho=stats.spearmanr(ctrp[c2col[nm]].loc[both],gdm[nm].loc[both]).statistic if len(both)>=20 else np.nan
    rows2.append(dict(drug=nm,rho=round(float(rho),3) if not np.isnan(rho) else None,
                      n_ext=len(ext_idx),int_spearman=round(float(rin),4),ext_spearman=round(float(rext),4)))
rg=pd.DataFrame(rows2); rg.to_csv(OUT/"R6_regression_per_drug.csv",index=False)
s=rg[["rho","ext_spearman"]].dropna(); spr=stats.spearmanr(s["rho"],s["ext_spearman"])
P(f"  drugs: {len(rg)}; median internal Spearman={rg['int_spearman'].median():.3f}, "
  f"median external Spearman={rg['ext_spearman'].median():.3f}")
P(f"  Spearman(concordance, external regression Spearman) = {spr.statistic:+.3f} (p={spr.pvalue:.1e})")
regsum=dict(n_drugs=int(len(rg)),median_int_spearman=round(float(rg['int_spearman'].median()),3),
            median_ext_spearman=round(float(rg['ext_spearman'].median()),3),
            spearman_rho_vs_ext=round(float(spr.statistic),3),spearman_p=float(spr.pvalue))
json.dump(dict(bidirectional=bsum,regression=regsum),open(OUT/"R6_robustness_summary.json","w"),indent=2)
open(OUT/"R6_robustness_log.txt","w").write("\n".join(log))
P(f"\n[done] -> {OUT}")
