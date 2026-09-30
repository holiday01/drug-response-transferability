#!/usr/bin/env python3
"""
Three-assay validation: does cross-assay concordance predict transferability
across THREE independent drug-sensitivity resources (CTRP, GDSC2, PRISM), not
just one pair? Leakage-free, seed 42; every value regenerated from public data.

For every ordered assay pair (train A -> test B) we compute per-drug internal and
external AUROC; per-drug concordance is the Spearman agreement of the two assays on
shared cell lines. We then test:
  (1) within each of the 6 ordered pairs, concordance -> external AUROC;
  (2) LEAVE-ONE-ASSAY-OUT: a drug's mean pairwise concordance (a drug-intrinsic
      reproducibility score) predicts its mean transfer to a HELD-OUT assay never
      used to compute that score.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="4"
import json, re, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
SEED=42; np.random.seed(SEED)
from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
CC=str(CCLE_DIR)
EXPR=f"{CC}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
CTRP=f"{CC}/Drug_sensitivity_AUC_(CTD^2)_subsetted.csv"
GDSC=f"{CC}/Drug_sensitivity_replicate-level_dose_(Sanger_GDSC2)_subsetted.csv"
PRISM=str(PRISM_AUC)
OUT=Path(str(RESULTS_DIR)); OUT.mkdir(exist_ok=True)
TOP_K,VAR_Q=200,0.50; MIN_LAB=120; MIN_CLASS=15
log=[]
def P(m=""): print(m,flush=True); log.append(str(m))
def bn(col,portal): return re.split(rf"\s*\({portal}:",col)[0].strip().upper()
def tert(s):
    s=s.dropna(); lo,hi=s.quantile(1/3),s.quantile(2/3)
    idx=s[s<=lo].index.append(s[s>=hi].index); y=pd.Series(0,index=idx); y.loc[s[s>=hi].index]=1
    return y
def selbin(X,y,k=TOP_K,vq=VAR_Q):
    v=X.var(0); keep=np.where(v>=np.quantile(v,vq))[0]; Xk=X[:,keep]; y=y.astype(float); n1=y.sum()
    if n1<2 or len(y)-n1<2: return keep[:k]
    m1=Xk[y==1].mean(0); m0=Xk[y==0].mean(0); sd=Xk.std(0); sd[sd==0]=np.nan
    r=np.nan_to_num(np.abs((m1-m0)/sd)); return keep[np.argsort(r)[::-1][:k]]

P(f"[{time.strftime('%H:%M:%S')}] loading CCLE expression ...")
expr=pd.read_csv(EXPR)
if "IsDefaultEntryForModel" in expr.columns:
    expr=expr[expr["IsDefaultEntryForModel"].astype(str).str.strip()=="Yes"]
META={"","SequencingID","ModelID","IsDefaultEntryForModel","ModelConditionID","IsDefaultEntryForMC"}
mc=[c for c in expr.columns if c in META or str(c).lower().startswith("unnamed")]
expr=(expr.dropna(subset=["ModelID"]).drop_duplicates("ModelID").set_index("ModelID")
        .drop(columns=[c for c in mc if c!="ModelID"],errors="ignore").astype(np.float32))
P(f"  CCLE {expr.shape}")

# ---- assay matrices: cell-line index, drug columns, low = sensitive
def load_ctrp():
    d=pd.read_csv(CTRP,index_col=0); mono=[c for c in d.columns if ":" not in c.split("(")[0]]; d=d[mono]
    d.columns=[bn(c,"CTRP") for c in d.columns]; return d.loc[:,~d.columns.duplicated()]
def load_gdsc():
    d=pd.read_csv(GDSC,index_col=0); d.index=d.index.astype(str); g={}
    for c in d.columns: g.setdefault(bn(c,"GDSC2"),[]).append(c)
    return pd.DataFrame({k:d[v].mean(1,skipna=True) for k,v in g.items()},index=d.index)
def load_prism():
    d=pd.read_csv(PRISM,usecols=["depmap_id","name","auc"]).dropna(subset=["depmap_id","name","auc"])
    d["name"]=d["name"].astype(str).str.upper().str.strip()
    m=d.groupby(["depmap_id","name"])["auc"].mean().unstack("name")
    return m
P(f"[{time.strftime('%H:%M:%S')}] loading assays ...")
A={"CTRP":load_ctrp(),"GDSC2":load_gdsc(),"PRISM":load_prism()}
for k,v in A.items(): P(f"  {k}: {v.shape[0]} cell lines x {v.shape[1]} drugs")

models={"Random Forest":lambda:RandomForestClassifier(n_estimators=300,n_jobs=4,random_state=SEED),
        "Logistic Regression":lambda:LogisticRegression(max_iter=2000,C=1.0,random_state=SEED)}
def prob(c,X): return c.predict_proba(X)[:,1]

pairs=[("CTRP","GDSC2"),("CTRP","PRISM"),("GDSC2","PRISM"),
       ("GDSC2","CTRP"),("PRISM","CTRP"),("PRISM","GDSC2")]
rows=[]
for src,dst in pairs:
    As,Ad=A[src],A[dst]
    shared_drugs=sorted(set(As.columns)&set(Ad.columns))
    P(f"\n[{time.strftime('%H:%M:%S')}] {src}->{dst}: {len(shared_drugs)} shared drugs")
    npair=0
    for nm in shared_drugs:
        ys=tert(As[nm]); ys=ys[ys.index.isin(expr.index)]
        yd_full=tert(Ad[nm]); yd_full=yd_full[yd_full.index.isin(expr.index)]
        if len(ys)<MIN_LAB or ys.nunique()<2: continue
        X=expr.loc[ys.index].to_numpy(); yv=ys.to_numpy()
        if min(np.bincount(yv))<20: continue
        Xtr,Xte,ytr,yte,itr,ite=train_test_split(X,yv,ys.index.to_numpy(),test_size=0.3,stratify=yv,random_state=SEED)
        feat=selbin(Xtr,ytr); sc=StandardScaler().fit(Xtr[:,feat])
        ext_idx=yd_full.index[~yd_full.index.isin(set(itr))]; yext=yd_full.loc[ext_idx]
        if yext.nunique()<2 or min(np.bincount(yext.to_numpy()))<MIN_CLASS: continue
        both=As[nm].dropna().index.intersection(Ad[nm].dropna().index)
        if len(both)<20: continue
        rho=stats.spearmanr(As[nm].loc[both],Ad[nm].loc[both]).statistic
        Xte_s=sc.transform(Xte[:,feat]); Xext_s=sc.transform(expr.loc[ext_idx].to_numpy()[:,feat])
        r=dict(pair=f"{src}->{dst}",src=src,dst=dst,drug=nm,rho=round(float(rho),3),n_ext=int(len(yext)))
        for mn,mk in models().items() if False else models.items():
            c=mk().fit(sc.transform(Xtr[:,feat]),ytr)
            r[f"int_{mn}"]=round(float(roc_auc_score(yte,prob(c,Xte_s))),4)
            r[f"ext_{mn}"]=round(float(roc_auc_score(yext.to_numpy(),prob(c,Xext_s))),4)
        rows.append(r); npair+=1
    P(f"  usable drugs: {npair}")

df=pd.DataFrame(rows); df.to_csv(OUT/"V_threeassay_per_pair.csv",index=False)

# ---- (1) per-pair concordance -> external AUROC
P(f"\n[{time.strftime('%H:%M:%S')}] === per-pair: concordance -> external AUROC (RF) ===")
pair_summ={}
for pr in df["pair"].unique():
    s=df[df.pair==pr][["rho","ext_Random Forest"]].dropna()
    sp=stats.spearmanr(s["rho"],s["ext_Random Forest"])
    pair_summ[pr]=dict(n=len(s),spearman=round(float(sp.statistic),3),p=float(sp.pvalue),
                       median_ext=round(float(s["ext_Random Forest"].median()),3),
                       median_rho=round(float(s["rho"].median()),3))
    P(f"  {pr:16s} n={len(s):2d}  Spearman(rho,ext)={sp.statistic:+.3f} (p={sp.pvalue:.1e})  "
      f"median ext={s['ext_Random Forest'].median():.3f}")
# pooled across all 6 pairs
s=df[["rho","ext_Random Forest"]].dropna()
pooled=stats.spearmanr(s["rho"],s["ext_Random Forest"])
P(f"  POOLED (all pairs, n={len(s)})  Spearman={pooled.statistic:+.3f} (p={pooled.pvalue:.1e})")

# ---- (2) leave-one-assay-out: drug-intrinsic reproducibility -> transfer to held-out assay
P(f"\n[{time.strftime('%H:%M:%S')}] === leave-one-assay-out (drug-intrinsic reproducibility) ===")
# per drug, per assay: mean external AUROC when that assay is the TEST target,
# vs mean concordance of the OTHER two assays (that never involve nothing... ) ->
# define reproducibility(drug) excluding transfers INTO the held-out assay.
loo=[]
for nm in df["drug"].unique():
    d=df[df.drug==nm]
    for held in ["CTRP","GDSC2","PRISM"]:
        into=d[d.dst==held]                                   # transfers INTO held-out assay
        if into.empty: continue
        ext_into=into["ext_Random Forest"].dropna()
        if ext_into.empty: continue
        # reproducibility score = concordance among pairs NOT involving the held-out assay
        other=d[(d.src!=held)&(d.dst!=held)]["rho"].dropna()
        if other.empty: continue
        loo.append(dict(drug=nm,held_out=held,reprod_score=float(other.mean()),
                        transfer_into=float(ext_into.mean())))
loo=pd.DataFrame(loo); loo.to_csv(OUT/"V_threeassay_loo.csv",index=False)
sp=stats.spearmanr(loo["reprod_score"],loo["transfer_into"])
P(f"  n={len(loo)}  Spearman(reproducibility of other pairs, transfer INTO held-out assay) "
  f"= {sp.statistic:+.3f} (p={sp.pvalue:.1e})")

meta=dict(seed=SEED,pairs=pair_summ,
          pooled_spearman=round(float(pooled.statistic),3),pooled_p=float(pooled.pvalue),
          pooled_n=int(len(s)),
          loo_spearman=round(float(sp.statistic),3),loo_p=float(sp.pvalue),loo_n=int(len(loo)),
          generated=time.strftime("%Y-%m-%d %H:%M:%S"))
(OUT/"V_threeassay_summary.json").write_text(json.dumps(meta,indent=2))
(OUT/"V_threeassay_log.txt").write_text("\n".join(log))
P(f"\n[done] -> {OUT}")
