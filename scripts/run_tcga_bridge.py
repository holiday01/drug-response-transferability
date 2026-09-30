#!/usr/bin/env python3
"""
R4 (translational capstone): do cell-line-trained models predict REAL PATIENT
drug response, and does cross-assay concordance predict which drugs translate?

Design (leakage-free, seed 42):
  * train on ALL CCLE cell lines with CTRP tertile-extreme labels (0 sensitive,
    1 resistant), leakage-free feature selection + standardisation on CCLE;
  * apply the fitted model UNCHANGED to TCGA patient tumors that received the
    same drug (GDC log-TPM protein-coding expression), on the shared gene space;
  * patient outcome = RECIST-aligned clinical response, binarised responder
    (CR+PR) vs non-responder (SD+PD); AUROC asks whether the model's predicted
    resistance probability ranks non-responders above responders.
  * domain shift handled by per-gene z-scoring WITHIN each domain separately.
Then correlate per-drug patient AUROC with the CTRP-GDSC2 concordance rho.

Every value regenerated from public data (CCLE/DepMap, CTRP, TCGA/GDC).
"""
from __future__ import annotations
import json, re, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

SEED = 42; np.random.seed(SEED)
from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
CCLE_DIR = str(CCLE_DIR)
EXPR = f"{CCLE_DIR}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
CTRP = f"{CCLE_DIR}/Drug_sensitivity_AUC_(CTD^2)_subsetted.csv"
GENE = f"{CCLE_DIR}/Gene.csv"
TCGA_EXPR = str(TCGA_EXPR)
TCGA_DRUG = str(TCGA_DRUG)
PERDRUG = str(RESULTS_DIR / "per_drug_external.csv")
OUT = Path(str(RESULTS_DIR)); OUT.mkdir(exist_ok=True)
TOP_K, VAR_Q = 200, 0.50
MIN_CTRP = 150          # CCLE labelled lines to train
MIN_PAT, MIN_CLASS = 25, 5   # TCGA patients per drug / per response class
N_BOOT = 2000
log = []
def P(m=""): print(m, flush=True); log.append(str(m))

RESP = {"Complete Response":0, "Partial Response":0,      # responder
        "Stable Disease":1, "Clinical Progressive Disease":1}  # non-responder
SYN = {  # TCGA drug name (upper) -> CTRP base name (upper)
    "5-FLUOROURACIL":"FLUOROURACIL", "5FU":"FLUOROURACIL", "FU":"FLUOROURACIL",
    "TAXOL":"PACLITAXEL", "TAXOTERE":"DOCETAXEL", "GEMZAR":"GEMCITABINE",
    "ADRIAMYCIN":"DOXORUBICIN", "CDDP":"CISPLATIN", "VP-16":"ETOPOSIDE",
    "VP16":"ETOPOSIDE",
}
def norm_drug(s):
    s = str(s).upper().strip()
    return SYN.get(s, s)

def base_name(col, portal):
    return re.split(rf"\s*\({portal}:", col)[0].strip().upper()

def tertile_binary(s):
    s = s.dropna(); lo, hi = s.quantile(1/3), s.quantile(2/3)
    idx = s[s<=lo].index.append(s[s>=hi].index)
    y = pd.Series(0, index=idx); y.loc[s[s>=hi].index] = 1
    return y

def select_features(Xtr, ytr, k=TOP_K, var_q=VAR_Q):
    v = Xtr.var(0); keep = np.where(v >= np.quantile(v, var_q))[0]
    Xk = Xtr[:, keep]; y = ytr.astype(float); n1=y.sum(); n0=len(y)-n1
    if n1<2 or n0<2: return keep[:k]
    m1=Xk[y==1].mean(0); m0=Xk[y==0].mean(0); sd=Xk.std(0); sd[sd==0]=np.nan
    r=np.nan_to_num(np.abs((m1-m0)/sd)*np.sqrt(n1/len(y)*(1-n1/len(y))))
    return keep[np.argsort(r)[::-1][:k]]

# ---------------- gene map ENSG<->symbol
P(f"[{time.strftime('%H:%M:%S')}] gene map ...")
g = pd.read_csv(GENE, usecols=["symbol","ensembl_gene_id"]).dropna()
ensg2sym = dict(zip(g.ensembl_gene_id.astype(str), g.symbol.astype(str)))

# ---------------- CCLE expression (symbol columns)
P(f"[{time.strftime('%H:%M:%S')}] loading CCLE expression ...")
expr = pd.read_csv(EXPR)
if "IsDefaultEntryForModel" in expr.columns:
    expr = expr[expr["IsDefaultEntryForModel"].astype(str).str.strip()=="Yes"]
META={"","SequencingID","ModelID","IsDefaultEntryForModel","ModelConditionID","IsDefaultEntryForMC"}
mc=[c for c in expr.columns if c in META or str(c).lower().startswith("unnamed")]
expr=(expr.dropna(subset=["ModelID"]).drop_duplicates("ModelID").set_index("ModelID")
        .drop(columns=[c for c in mc if c!="ModelID"], errors="ignore"))
expr.columns=[re.split(r"\s*\(",c)[0].strip() for c in expr.columns]   # symbol only
expr=expr.loc[:, ~expr.columns.duplicated()].astype(np.float32)
P(f"  CCLE {expr.shape}")

# ---------------- TCGA expression header -> common genes
P(f"[{time.strftime('%H:%M:%S')}] mapping TCGA genes ...")
thead=pd.read_csv(TCGA_EXPR, nrows=0)
meta_cols=["sample_id","barcode","cancer_code","sample_type"]
ensg_cols=[c for c in thead.columns if c.startswith("ENSG")]
ensg_base={c: c.split(".")[0] for c in ensg_cols}
ensg_sym={c: ensg2sym.get(ensg_base[c]) for c in ensg_cols}
common_cols=[c for c in ensg_cols if ensg_sym[c] in set(expr.columns)]
# unique symbol mapping
seen=set(); keep_cols=[]
for c in common_cols:
    s=ensg_sym[c]
    if s not in seen: seen.add(s); keep_cols.append(c)
common_sym=[ensg_sym[c] for c in keep_cols]
P(f"  common protein-coding genes CCLE∩TCGA: {len(keep_cols)}")

# ---------------- load TCGA expression (meta + common genes)
P(f"[{time.strftime('%H:%M:%S')}] loading TCGA expression (common genes) ...")
tcga=pd.read_csv(TCGA_EXPR, usecols=meta_cols+keep_cols)
tcga=tcga[tcga.sample_type=="Primary Tumor"].copy()
tcga=tcga.rename(columns={c:ensg_sym[c] for c in keep_cols})
tcga=tcga.drop_duplicates("barcode").set_index("barcode")
Xtcga_all=tcga[common_sym].astype(np.float32)
P(f"  TCGA primary-tumor samples: {Xtcga_all.shape[0]}")

# ---------------- TCGA drug response
dr=pd.read_csv(TCGA_DRUG)
dr["drug_u"]=dr["drug_name"].map(norm_drug)
dr["y"]=dr["measure_of_response"].map(RESP)
dr=dr.dropna(subset=["y"]); dr["y"]=dr["y"].astype(int)

# ---------------- CTRP
ctrp=pd.read_csv(CTRP, index_col=0)
mono=[c for c in ctrp.columns if ":" not in c.split("(")[0]]; ctrp=ctrp[mono]
c2col={}
for c in ctrp.columns:
    nm=base_name(c,"CTRP")
    if nm not in c2col or ctrp[c].notna().sum()>ctrp[c2col[nm]].notna().sum(): c2col[nm]=c

# concordance table
conc=pd.read_csv(PERDRUG); conc=conc.set_index("drug")["spearman_ctrp_gdsc2"].astype(float).to_dict()

def zscore(M):
    mu=M.mean(0); sd=M.std(0); sd[sd==0]=1.0
    return (M-mu)/sd

def quantile_homogenize(Xref, Xnew):
    """Map each gene of Xnew (patients) onto the marginal distribution of the same
    gene in Xref (cell lines), a per-gene quantile normalisation (Geeleher-style
    cross-domain homogenisation). Returns Xnew mapped into the Xref feature space."""
    out=np.empty_like(Xnew, dtype=np.float64)
    n=Xnew.shape[0]
    for j in range(Xnew.shape[1]):
        ref=np.sort(Xref[:,j].astype(np.float64))
        r=stats.rankdata(Xnew[:,j], method="average")/(n+1)   # percentile in Xnew
        out[:,j]=np.interp(r, np.linspace(0,1,len(ref)), ref) # value at that percentile of ref
    return out

MODELS={"Logistic Regression":LogisticRegression(max_iter=2000,C=1.0,random_state=SEED),
        "Random Forest":RandomForestClassifier(n_estimators=400,n_jobs=-1,random_state=SEED),
        "SVM (RBF)":SVC(kernel="rbf",C=1.0,probability=True,random_state=SEED)}

def boot_auc_ci(y,s,n=N_BOOT):
    y=np.asarray(y); s=np.asarray(s); rng=np.random.default_rng(SEED); out=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(set(y[i]))<2: continue
        out.append(roc_auc_score(y[i],s[i]))
    return (np.percentile(out,2.5),np.percentile(out,97.5)) if out else (np.nan,np.nan)

# ---------------- per drug: train CCLE, predict TCGA patients
P(f"\n[{time.strftime('%H:%M:%S')}] === TCGA patient bridge ===")
rows=[]
drugs_tcga=sorted(dr["drug_u"].unique())
for nm in drugs_tcga:
    if nm not in c2col: continue
    sub=dr[dr.drug_u==nm]
    pats=[b for b in sub["bcr_patient_barcode"].unique() if b in Xtcga_all.index]
    if len(pats)<MIN_PAT: continue
    yp=sub.drop_duplicates("bcr_patient_barcode").set_index("bcr_patient_barcode").loc[
        [p for p in pats],"y"].values
    if min(np.bincount(yp))<MIN_CLASS: continue
    # CCLE training labels
    yc_all=tertile_binary(ctrp[c2col[nm]])
    yc_all=yc_all[yc_all.index.isin(expr.index)]
    if len(yc_all)<MIN_CTRP or yc_all.nunique()<2: continue
    Xc=zscore(expr.loc[yc_all.index, common_sym].to_numpy())   # per-gene z within CCLE
    yc=yc_all.to_numpy()
    feat=select_features(Xc, yc)
    sc=StandardScaler().fit(Xc[:,feat]); Xc_s=sc.transform(Xc[:,feat])
    # TCGA patients: per-gene z within the TCGA domain (removes cross-domain location/scale)
    Xp=zscore(Xtcga_all.loc[pats, :].to_numpy())
    Xp_s=sc.transform(Xp[:,feat])
    row=dict(drug=nm, n_pat=len(pats), n_nonresp=int(yp.sum()), n_resp=int((yp==0).sum()),
             n_ctrp=int(len(yc)), rho=conc.get(nm, np.nan))
    aucs=[]
    for mnm,clf in MODELS.items():
        clf.fit(Xc_s, yc)
        p_resist=clf.predict_proba(Xp_s)[:,1]     # higher = more resistant
        auc=roc_auc_score(yp, p_resist)           # patient y: 1=nonresponder
        row[f"patAUROC_{mnm}"]=round(float(auc),4); aucs.append(auc)
    row["patAUROC_mean"]=round(float(np.mean(aucs)),4)
    lo,hi=boot_auc_ci(yp, MODELS["Random Forest"].fit(Xc_s,yc).predict_proba(Xp_s)[:,1])
    row["patAUROC_RF_ci"]=f"{lo:.3f}-{hi:.3f}"
    rows.append(row)
    P(f"  {nm:16s} n_pat={len(pats):3d} ({int((yp==0).sum())}R/{int(yp.sum())}NR) "
      f"rho={row['rho']}  patAUROC mean={row['patAUROC_mean']:.3f} "
      f"RF={row['patAUROC_Random Forest']:.3f} [{row['patAUROC_RF_ci']}]")

R=pd.DataFrame(rows)
R.to_csv(OUT/"R4_tcga_bridge_per_drug.csv", index=False)
P(f"\n  drugs with a usable TCGA test set: {len(R)}")

# ---------------- does concordance predict patient translation?
have=R.dropna(subset=["rho"])
summ=dict(n_drugs=int(len(R)), n_drugs_with_rho=int(len(have)),
          median_patAUROC_mean=round(float(R["patAUROC_mean"].median()),3))
if len(have)>=4:
    spm=stats.spearmanr(have["rho"], have["patAUROC_mean"])
    spr=stats.spearmanr(have["rho"], have["patAUROC_Random Forest"])
    summ["spearman_rho_vs_patAUROCmean"]=round(float(spm.statistic),3)
    summ["spearman_rho_vs_patAUROCmean_p"]=float(spm.pvalue)
    summ["spearman_rho_vs_patAUROC_RF"]=round(float(spr.statistic),3)
    P(f"\n  Spearman(concordance, patient AUROC mean) = {spm.statistic:+.3f} (p={spm.pvalue:.3f})")
# above-chance count
n_above=int((R["patAUROC_mean"]>0.5).sum())
summ["n_above_chance"]=n_above; summ["n_total"]=int(len(R))
P(f"  drugs with patient AUROC > 0.5 (mean over models): {n_above}/{len(R)}")
summ["seed"]=SEED; summ["generated"]=time.strftime("%Y-%m-%d %H:%M:%S")
(OUT/"R4_tcga_bridge_summary.json").write_text(json.dumps(summ,indent=2))
(OUT/"R4_tcga_bridge_log.txt").write_text("\n".join(log))
P(f"\n[done] -> {OUT}")
