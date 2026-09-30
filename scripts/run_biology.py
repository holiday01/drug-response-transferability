#!/usr/bin/env python3
"""
Biological validation of concordance -> transferability:
  (1) MOA class vs concordance/transfer (targeted / cytotoxic / epigenetic / other);
  (2) known-biomarker recovery: rank of a drug's canonical target gene among the
      point-biserial features, concordant vs discordant drugs;
  (3) cross-assay feature overlap: Jaccard of top genes selected on CTRP vs GDSC2
      labels, related to concordance (why concordant drugs transfer).
Seed 42; every value regenerated from public data.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="4"
import re,json,warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
R=str(RESULTS_DIR)
CC=str(CCLE_DIR)
EXPR=f"{CC}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
CTRP=f"{CC}/Drug_sensitivity_AUC_(CTD^2)_subsetted.csv"
GDSC=f"{CC}/Drug_sensitivity_replicate-level_dose_(Sanger_GDSC2)_subsetted.csv"
CATALOG=str(DRUG_CATALOG)
SEED=42; TOPK=200; VAR_Q=0.50
log=[]; P=lambda m="":(print(m,flush=True),log.append(str(m)))
def bn(c,p): return re.split(rf"\s*\({p}:",c)[0].strip().upper()

# canonical drug -> known biomarker gene(s) (well-established pharmacogenomic markers)
BIOMARK={
 "TRAMETINIB":["MAP2K1","BRAF","NRAS"],"SELUMETINIB":["MAP2K1","BRAF","NRAS"],
 "PD 0325901":["MAP2K1","BRAF","NRAS"],"PD0325901":["MAP2K1","BRAF","NRAS"],
 "ERLOTINIB":["EGFR"],"GEFITINIB":["EGFR"],"AFATINIB":["EGFR","ERBB2"],
 "LAPATINIB":["ERBB2","EGFR"],"CRIZOTINIB":["ALK","MET","ROS1"],
 "DASATINIB":["ABL1","SRC","KIT"],"NILOTINIB":["ABL1","KIT"],"IMATINIB":["ABL1","KIT","PDGFRA"],
 "OLAPARIB":["BRCA1","BRCA2","PARP1"],"DABRAFENIB":["BRAF"],"PLX 4720":["BRAF"],"PLX4720":["BRAF"],
 "ALPELISIB":["PIK3CA"],"PICTILISIB":["PIK3CA"],"BEZ235":["PIK3CA","MTOR"],
 "VENETOCLAX":["BCL2"],"NAVITOCLAX":["BCL2","BCL2L1"],"MK 2206":["AKT1","AKT2"],
 "NUTLIN-3":["MDM2","TP53"],"RUXOLITINIB":["JAK1","JAK2"],"RAPAMYCIN":["MTOR"],
 "PACLITAXEL":["TUBB"],"DOCETAXEL":["TUBB"],"VINCRISTINE":["TUBB"],
}
def moa_class(moa,target):
    m=str(moa).lower()
    if any(k in m for k in ["hdac","dnmt","bromodomain","bet ","ezh2","histone","demethylase","epigenetic"]): return "epigenetic"
    if any(k in m for k in ["topoisomerase","tubulin","microtubule","dna ","alkylat","antimetabolite","intercalat","nucleoside","cytotoxic","platinum"]): return "cytotoxic"
    if "inhibitor" in m or "antagonist" in m or "kinase" in m: return "targeted"
    return "other"

# ---- load
P("loading expression ...")
expr=pd.read_csv(EXPR)
if "IsDefaultEntryForModel" in expr.columns:
    expr=expr[expr["IsDefaultEntryForModel"].astype(str).str.strip()=="Yes"]
META={"","SequencingID","ModelID","IsDefaultEntryForModel","ModelConditionID","IsDefaultEntryForMC"}
mc=[c for c in expr.columns if c in META or str(c).lower().startswith("unnamed")]
expr=(expr.dropna(subset=["ModelID"]).drop_duplicates("ModelID").set_index("ModelID")
        .drop(columns=[c for c in mc if c!="ModelID"],errors="ignore"))
expr.columns=[re.split(r"\s*\(",c)[0].strip() for c in expr.columns]
expr=expr.loc[:,~expr.columns.duplicated()].astype(np.float32)
genes=np.array(expr.columns); gidx={g:i for i,g in enumerate(genes)}
ctrp=pd.read_csv(CTRP,index_col=0); mono=[c for c in ctrp.columns if ":" not in c.split("(")[0]]; ctrp=ctrp[mono]
c2={}
for c in ctrp.columns:
    nm=bn(c,"CTRP")
    if nm not in c2 or ctrp[c].notna().sum()>ctrp[c2[nm]].notna().sum(): c2[nm]=c
gd=pd.read_csv(GDSC,index_col=0); gd.index=gd.index.astype(str); gg={}
for c in gd.columns: gg.setdefault(bn(c,"GDSC2"),[]).append(c)
gdm=pd.DataFrame({k:gd[v].mean(1,skipna=True) for k,v in gg.items()},index=gd.index)
ext=pd.read_csv(f"{R}/per_drug_external.csv"); ext=ext[ext.n_ext>0].copy()
ext["rho"]=ext["spearman_ctrp_gdsc2"].astype(float)
cat=pd.read_parquet(CATALOG); cat["drug_u"]=cat["drug"].astype(str).str.upper().str.strip()
moa_map=cat.drop_duplicates("drug_u").set_index("drug_u")[["moa","target"]].to_dict("index")

def tert(s):
    s=s.dropna(); lo,hi=s.quantile(1/3),s.quantile(2/3)
    idx=s[s<=lo].index.append(s[s>=hi].index); y=pd.Series(0,index=idx); y.loc[s[s>=hi].index]=1
    return y
def pbis_rank(vals):
    """point-biserial |r| for every gene given a tertile label series on shared cells; returns ranks (1=top)."""
    y=vals[vals.index.isin(expr.index)]
    X=expr.loc[y.index].to_numpy(); yv=y.to_numpy().astype(float); n1=yv.sum()
    if n1<3 or len(yv)-n1<3: return None
    m1=X[yv==1].mean(0); m0=X[yv==0].mean(0); sd=X.std(0); sd[sd==0]=np.nan
    r=np.nan_to_num(np.abs((m1-m0)/sd))
    order=np.argsort(r)[::-1]; rankarr=np.empty(len(r),int); rankarr[order]=np.arange(1,len(r)+1)
    return r, rankarr

# ---- (1) MOA class + (2) biomarker recovery + (3) feature overlap, per drug
rows=[]
for _,rr in ext.iterrows():
    nm=rr["drug"]; rho=rr["rho"]
    yc=tert(ctrp[c2[nm]]); res_c=pbis_rank(yc)
    yg=tert(gdm[nm]) if nm in gdm.columns else pd.Series(dtype=float); res_g=pbis_rank(yg) if len(yg) else None
    if res_c is None: continue
    r_c,rank_c=res_c
    # top-K gene sets
    topC=set(genes[np.argsort(r_c)[::-1][:TOPK]])
    jac=np.nan
    if res_g is not None:
        r_g,_=res_g; topG=set(genes[np.argsort(r_g)[::-1][:TOPK]])
        jac=len(topC&topG)/len(topC|topG)
    mo=moa_map.get(nm,{}); cls=moa_class(mo.get("moa"),mo.get("target"))
    # biomarker recovery: best (smallest) rank of any known target gene
    bmrank=np.nan; bm=None
    if nm in BIOMARK:
        rs=[rank_c[gidx[g]] for g in BIOMARK[nm] if g in gidx]
        if rs: bmrank=int(min(rs)); bm=BIOMARK[nm]
    rows.append(dict(drug=nm,rho=rho,ext_RF=rr["ext_Random Forest"],moa=mo.get("moa"),
                     moa_class=cls,feature_jaccard=None if np.isnan(jac) else round(float(jac),4),
                     biomarker_genes=None if bm is None else ";".join(bm),
                     biomarker_rank=None if np.isnan(bmrank) else bmrank))
B=pd.DataFrame(rows); B.to_csv(f"{R}/V_biology_per_drug.csv",index=False)

# --- summarise (1) MOA
P("\n=== MOA class vs concordance / transfer ===")
moa_summ={}
for cls in ["targeted","cytotoxic","epigenetic","other"]:
    g=B[B.moa_class==cls]
    if len(g)<3: continue
    moa_summ[cls]=dict(n=len(g),median_rho=round(float(g["rho"].median()),3),
                       median_ext=round(float(g["ext_RF"].median()),3))
    P(f"  {cls:11s} n={len(g):2d}  median rho={g['rho'].median():.3f}  median ext AUROC={g['ext_RF'].median():.3f}")
kw=stats.kruskal(*[B[B.moa_class==c]["rho"].dropna() for c in ["targeted","cytotoxic","epigenetic","other"] if (B.moa_class==c).sum()>=3])
P(f"  Kruskal-Wallis (concordance across MOA classes): H={kw.statistic:.2f}, p={kw.pvalue:.2e}")

# --- (3) feature overlap vs concordance
P("\n=== cross-assay feature overlap (Jaccard of top-200 genes) vs concordance ===")
fj=B[["rho","feature_jaccard"]].dropna()
spj=stats.spearmanr(fj["rho"],fj["feature_jaccard"])
P(f"  n={len(fj)}  Spearman(concordance, feature Jaccard) = {spj.statistic:+.3f} (p={spj.pvalue:.1e}); "
  f"median Jaccard={fj['feature_jaccard'].median():.3f}")

# --- (2) biomarker recovery vs concordance
P("\n=== known-biomarker recovery (rank of target gene; lower=better) ===")
bm=B.dropna(subset=["biomarker_rank"]).copy(); bm["biomarker_rank"]=bm["biomarker_rank"].astype(int)
P(f"  drugs with a curated biomarker: {len(bm)}")
in_top=(bm["biomarker_rank"]<=TOPK).sum()
P(f"  target gene in top-{TOPK} features: {in_top}/{len(bm)} drugs")
spb=stats.spearmanr(bm["rho"],bm["biomarker_rank"])
P(f"  Spearman(concordance, biomarker rank) = {spb.statistic:+.3f} (p={spb.pvalue:.2e})  "
  f"(negative = concordant drugs rank their target higher)")
for _,r in bm.sort_values("rho",ascending=False).iterrows():
    P(f"    {r['drug']:16s} rho={r['rho']:+.2f}  {r['biomarker_genes']:20s} best rank={int(r['biomarker_rank'])}")

meta=dict(moa=moa_summ, moa_kruskal_p=float(kw.pvalue),
          feature_jaccard_spearman=round(float(spj.statistic),3),feature_jaccard_p=float(spj.pvalue),
          median_jaccard=round(float(fj["feature_jaccard"].median()),3),
          biomarker_in_topk=int(in_top),biomarker_n=int(len(bm)),
          biomarker_rank_spearman=round(float(spb.statistic),3),biomarker_rank_p=float(spb.pvalue),topk=TOPK)
json.dump(meta,open(f"{R}/V_biology_summary.json","w"),indent=2)
open(f"{R}/V_biology_log.txt","w").write("\n".join(log))
P("\n[done]")
