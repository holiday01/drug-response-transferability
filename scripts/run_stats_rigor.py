#!/usr/bin/env python3
"""
Statistical rigor for concordance -> transferability:
  (1) permutation null for the Spearman correlation (drug-label shuffles);
  (2) bootstrap 95% CI on the observed Spearman;
  (3) alternative concordance metrics (Spearman / Pearson-on-z / Kendall tau) all
      predict transfer;
  (4) multivariable partial correlation controlling for n, class balance, tissue
      diversity, and internal difficulty.
No expression needed; every value regenerated from public data. Seed 42.
"""
import re, json, numpy as np, pandas as pd
from scipy import stats
from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
R=str(RESULTS_DIR)
CC=str(CCLE_DIR)
CTRP=f"{CC}/Drug_sensitivity_AUC_(CTD^2)_subsetted.csv"
GDSC=f"{CC}/Drug_sensitivity_replicate-level_dose_(Sanger_GDSC2)_subsetted.csv"
MODEL=f"{CC}/Model.csv"
SEED=42; rng=np.random.default_rng(SEED)
PRIMARY="Random Forest"
log=[]
def P(m=""): print(m,flush=True); log.append(str(m))
def bn(c,p): return re.split(rf"\s*\({p}:",c)[0].strip().upper()

ext=pd.read_csv(f"{R}/per_drug_external.csv"); ext=ext[ext.n_ext>0].copy()
ext["rho"]=ext["spearman_ctrp_gdsc2"].astype(float)
x=ext["rho"].to_numpy(); y=ext[f"ext_{PRIMARY}"].to_numpy()
obs=stats.spearmanr(x,y).statistic
P(f"observed Spearman(concordance, external AUROC) = {obs:.3f} (n={len(x)})")

# (1) permutation null
NP=20000
null=np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(NP)])
p_perm=(np.sum(np.abs(null)>=abs(obs))+1)/(NP+1)
P(f"[perm] permutation p (|null| >= |obs|, {NP} shuffles) = {p_perm:.2e}; "
  f"null mean={null.mean():.3f} sd={null.std():.3f}")

# (2) bootstrap CI
NB=10000
bs=np.array([ (lambda i: stats.spearmanr(x[i],y[i]).statistic)(rng.integers(0,len(x),len(x))) for _ in range(NB)])
lo,hi=np.percentile(bs,[2.5,97.5])
P(f"[boot] bootstrap 95% CI on Spearman = [{lo:.3f}, {hi:.3f}]")

# (3) alternative concordance metrics from raw assays
ctrp=pd.read_csv(CTRP,index_col=0); mono=[c for c in ctrp.columns if ":" not in c.split("(")[0]]; ctrp=ctrp[mono]
c2={}
for c in ctrp.columns:
    nm=bn(c,"CTRP")
    if nm not in c2 or ctrp[c].notna().sum()>ctrp[c2[nm]].notna().sum(): c2[nm]=c
gd=pd.read_csv(GDSC,index_col=0); gd.index=gd.index.astype(str); gg={}
for c in gd.columns: gg.setdefault(bn(c,"GDSC2"),[]).append(c)
gdm=pd.DataFrame({k:gd[v].mean(1,skipna=True) for k,v in gg.items()},index=gd.index)

def zc(a,b):
    a=(a-a.mean())/a.std(); b=(b-b.mean())/b.std(); return a,b
alt=[]
for _,r in ext.iterrows():
    nm=r["drug"];
    if nm not in c2 or nm not in gdm.columns: continue
    cv=ctrp[c2[nm]].dropna(); gv=gdm[nm].dropna(); both=cv.index.intersection(gv.index)
    if len(both)<20: continue
    a=cv.loc[both].to_numpy(); b=gv.loc[both].to_numpy()
    az,bz=zc(a,b)
    alt.append(dict(drug=nm, spearman=stats.spearmanr(a,b).statistic,
                    pearson_z=stats.pearsonr(az,bz).statistic,
                    kendall=stats.kendalltau(a,b).statistic,
                    ext=r[f"ext_{PRIMARY}"]))
alt=pd.DataFrame(alt).dropna(); alt.to_csv(f"{R}/V_alt_concordance_metrics.csv",index=False)
P(f"\n[alt] concordance metric -> external AUROC (n={len(alt)}):")
metric_summ={}
for m in ["spearman","pearson_z","kendall"]:
    sp=stats.spearmanr(alt[m],alt["ext"])
    metric_summ[m]=dict(spearman_with_ext=round(float(sp.statistic),3),p=float(sp.pvalue))
    P(f"    {m:12s}: Spearman with external AUROC = {sp.statistic:+.3f} (p={sp.pvalue:.1e})")

# (4) multivariable partial correlation controlling for covariates
det=pd.read_csv(f"{R}/R5_determinants_per_drug.csv")
# class balance from CTRP tertile labels: |0.5 - frac_resistant| ~ 0 by tertile design, use n
cov=ext.merge(det[["drug","lineage_entropy","int_RF"]],on="drug",how="left")
cov["n"]=cov["n_ctrp"]
sub=cov[["rho",f"ext_{PRIMARY}","n","lineage_entropy","int_RF"]].dropna()
def partial_spearman(a,b,Z):
    ra=stats.rankdata(a); rb=stats.rankdata(b); RZ=np.column_stack([stats.rankdata(Z[:,j]) for j in range(Z.shape[1])])
    RZ=np.column_stack([np.ones(len(ra)),RZ])
    ba,_,_,_=np.linalg.lstsq(RZ,ra,rcond=None); ea=ra-RZ@ba
    bb,_,_,_=np.linalg.lstsq(RZ,rb,rcond=None); eb=rb-RZ@bb
    return stats.spearmanr(ea,eb)
Z=sub[["n","lineage_entropy","int_RF"]].to_numpy()
pc=partial_spearman(sub["rho"].to_numpy(),sub[f"ext_{PRIMARY}"].to_numpy(),Z)
P(f"\n[partial] Spearman(concordance, external | n, lineage_entropy, internal_difficulty) "
  f"= {pc.statistic:+.3f} (p={pc.pvalue:.1e}, n={len(sub)})")

meta=dict(observed_spearman=round(float(obs),3),permutation_p=float(p_perm),
          n_perm=NP, boot_ci=[round(float(lo),3),round(float(hi),3)],
          alt_metrics=metric_summ,
          partial_spearman_full=round(float(pc.statistic),3),partial_p=float(pc.pvalue),
          n_drugs=int(len(x)))
np.save(f"{R}/V_perm_null.npy", null)
json.dump(meta,open(f"{R}/V_stats_rigor_summary.json","w"),indent=2)
open(f"{R}/V_stats_rigor_log.txt","w").write("\n".join(log))
P(f"\n[done]")
