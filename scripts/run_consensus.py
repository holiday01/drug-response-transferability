#!/usr/bin/env python3
"""
R3 (positive METHOD test): does concordance-based CONSENSUS label denoising
improve recovery of the reproducible drug-response phenotype?

Non-leaky design, per drug, on cell lines with BOTH CTRP and GDSC2 sensitivity
and expression:
  * split shared cell lines TRAIN(70%)/TEST(30%), stratified on CTRP label, seed 42;
    TEST cells are held out from every training scheme.
  * training schemes (features + scaler fit on TRAIN cells only, always):
      S_full  : CTRP tertile labels, ALL CTRP-extreme train cells
      C       : train only on train cells where CTRP & GDSC2 AGREE (both sensitive
                or both resistant) -> denoised labels, fewer cells
      S_down  : CTRP labels, RANDOM train subset matched to C's size (controls for
                the smaller training set, isolating the effect of label QUALITY)
  * evaluation targets on HELD-OUT TEST cells:
      consensus-truth : test cells where CTRP & GDSC2 agree (the reproducible signal)
      gdsc2           : test cells' independent GDSC2 label (external generalization)
      ctrp            : test cells' CTRP label (does denoising hurt same-assay pred?)

Positive result = C beats S_full AND S_down on the consensus-truth target.
Seed fixed; every value regenerated from public data.
"""
from __future__ import annotations
import json, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
BASE = str(CCLE_DIR)
EXPR = f"{BASE}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
CTRP = f"{BASE}/Drug_sensitivity_AUC_(CTD^2)_subsetted.csv"
GDSC = f"{BASE}/Drug_sensitivity_replicate-level_dose_(Sanger_GDSC2)_subsetted.csv"
OUT = Path(str(RESULTS_DIR))
TOP_K, VAR_Q = 200, 0.50
MIN_SHARED = 120          # cell lines with both assays + expression
MIN_AGREE_CLASS = 12      # min concordant train cells per class for scheme C
MIN_TEST_CLASS = 8
N_BOOT = 2000
log = []
def P(m=""):
    print(m, flush=True); log.append(str(m))

def base_name(col, portal):
    return re.split(rf"\s*\({portal}:", col)[0].strip().upper()

def tertile_series(s):
    """return dict index-> {0 sensitive,1 resistant}; middle omitted."""
    s = s.dropna(); lo, hi = s.quantile(1/3), s.quantile(2/3)
    lab = {}
    for i, v in s.items():
        if v <= lo: lab[i] = 0
        elif v >= hi: lab[i] = 1
    return lab

def select_features(Xtr, ytr, k=TOP_K, var_q=VAR_Q):
    v = Xtr.var(axis=0); keep = np.where(v >= np.quantile(v, var_q))[0]
    Xk = Xtr[:, keep]; y = ytr.astype(float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 < 2 or n0 < 2: return keep[:k]
    m1 = Xk[y == 1].mean(0); m0 = Xk[y == 0].mean(0)
    sd = Xk.std(0); sd[sd == 0] = np.nan; p = n1/len(y)
    r = np.nan_to_num(np.abs((m1-m0)/sd)*np.sqrt(p*(1-p)))
    return keep[np.argsort(r)[::-1][:k]]

def newmodels():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
    }

def prob(clf, X): return clf.predict_proba(X)[:, 1]

# ---------------------------------------------------------------- load
P(f"[{time.strftime('%H:%M:%S')}] loading expression ...")
expr = pd.read_csv(EXPR)
if "IsDefaultEntryForModel" in expr.columns:
    expr = expr[expr["IsDefaultEntryForModel"].astype(str).str.strip() == "Yes"]
META = {"", "SequencingID", "ModelID", "IsDefaultEntryForModel", "ModelConditionID", "IsDefaultEntryForMC"}
mc = [c for c in expr.columns if c in META or str(c).lower().startswith("unnamed")]
expr = (expr.dropna(subset=["ModelID"]).drop_duplicates("ModelID").set_index("ModelID")
        .drop(columns=[c for c in mc if c != "ModelID"], errors="ignore").astype(np.float32))
P(f"  expression {expr.shape}")
ctrp = pd.read_csv(CTRP, index_col=0)
mono = [c for c in ctrp.columns if ":" not in c.split("(")[0]]; ctrp = ctrp[mono]
c2col = {}
for c in ctrp.columns:
    nm = base_name(c, "CTRP")
    if nm not in c2col or ctrp[c].notna().sum() > ctrp[c2col[nm]].notna().sum(): c2col[nm] = c
gd = pd.read_csv(GDSC, index_col=0); gd.index = gd.index.astype(str)
gg = {}
for c in gd.columns: gg.setdefault(base_name(c, "GDSC2"), []).append(c)
gdm = pd.DataFrame({nm: gd[cols].mean(1, skipna=True) for nm, cols in gg.items()}, index=gd.index)
shared_drugs = sorted(set(c2col) & set(gdm.columns))
P(f"  shared drugs {len(shared_drugs)}")

rows = []
for nm in shared_drugs:
    cvals = ctrp[c2col[nm]].dropna(); gvals = gdm[nm].dropna()
    both = cvals.index.intersection(gvals.index).intersection(expr.index)
    if len(both) < MIN_SHARED: continue
    cl = tertile_series(cvals.loc[both]); gl = tertile_series(gvals.loc[both])
    # cells with a CTRP extreme label (basis for S schemes)
    ctrp_cells = [i for i in both if i in cl]
    if len(ctrp_cells) < 60: continue
    yC = np.array([cl[i] for i in ctrp_cells])
    if min(np.bincount(yC)) < 20: continue
    idx_tr, idx_te = train_test_split(np.array(ctrp_cells), test_size=0.30,
                                      stratify=yC, random_state=SEED)
    tr_set, te_set = set(idx_tr), set(idx_te)
    # ---- training label sets
    # S_full: all CTRP-extreme train cells
    S_cells = list(idx_tr); S_y = np.array([cl[i] for i in S_cells])
    # C: train cells where CTRP & GDSC2 agree
    C_cells = [i for i in idx_tr if i in gl and cl[i] == gl[i]]
    C_y = np.array([cl[i] for i in C_cells]) if C_cells else np.array([])
    if len(C_cells) < 2*MIN_AGREE_CLASS or C_y.size == 0 or min(np.bincount(C_y)) < MIN_AGREE_CLASS:
        continue
    # S_down: random CTRP-label subset matched to |C|
    rs = np.random.RandomState(SEED)
    perm = rs.permutation(len(S_cells))[:len(C_cells)]
    D_cells = [S_cells[j] for j in perm]; D_y = np.array([cl[i] for i in D_cells])
    if min(np.bincount(D_y)) < 3:  # keep both classes
        # fall back to stratified downsample
        pos = [i for i in S_cells if cl[i]==1]; neg=[i for i in S_cells if cl[i]==0]
        k = len(C_cells)//2
        D_cells = list(rs.choice(pos, min(k,len(pos)), False)) + list(rs.choice(neg, min(k,len(neg)), False))
        D_y = np.array([cl[i] for i in D_cells])
    # ---- test targets on held-out cells
    cons_te = [i for i in idx_te if i in gl and cl[i] == gl[i]]      # consensus-truth
    cons_y = np.array([cl[i] for i in cons_te]) if cons_te else np.array([])
    gd_te = [i for i in idx_te if i in gl]; gd_y = np.array([gl[i] for i in gd_te]) if gd_te else np.array([])
    ctrp_te = list(idx_te); ctrp_y = np.array([cl[i] for i in ctrp_te])
    def ok(y): return y.size and len(set(y)) == 2 and min(np.bincount(y)) >= MIN_TEST_CLASS
    if not ok(cons_y): continue

    def fit_eval(cells, y_tr):
        Xtr = expr.loc[cells].to_numpy()
        feat = select_features(Xtr, y_tr)
        sc = StandardScaler().fit(Xtr[:, feat])
        Xtr_s = sc.transform(Xtr[:, feat])
        out = {}
        for mnm, clf in newmodels().items():
            clf.fit(Xtr_s, y_tr)
            for tgt, cells_t, y_t in [("cons", cons_te, cons_y), ("gdsc2", gd_te, gd_y), ("ctrp", ctrp_te, ctrp_y)]:
                if not ok(y_t): out[f"{mnm}|{tgt}"] = np.nan; continue
                Xt = sc.transform(expr.loc[cells_t].to_numpy()[:, feat])
                out[f"{mnm}|{tgt}"] = roc_auc_score(y_t, prob(clf, Xt))
        return out
    rS = fit_eval(S_cells, S_y)
    rC = fit_eval(C_cells, C_y)
    rD = fit_eval(D_cells, D_y)
    rho = stats.spearmanr(cvals.loc[both], gvals.loc[both]).statistic
    row = dict(drug=nm, n_shared=len(both), n_train_full=len(S_cells),
               n_train_cons=len(C_cells), rho=round(float(rho), 3),
               n_test_cons=len(cons_te))
    for mnm in newmodels():
        for tgt in ["cons", "gdsc2", "ctrp"]:
            row[f"S_{mnm}|{tgt}"] = rS.get(f"{mnm}|{tgt}")
            row[f"C_{mnm}|{tgt}"] = rC.get(f"{mnm}|{tgt}")
            row[f"D_{mnm}|{tgt}"] = rD.get(f"{mnm}|{tgt}")
    rows.append(row)
    P(f"  {nm:20s} nsh={len(both):3d} ncons_tr={len(C_cells):3d} rho={rho:+.2f} "
      f"| cons RF S={rS.get('Random Forest|cons',float('nan')):.3f} "
      f"C={rC.get('Random Forest|cons',float('nan')):.3f} "
      f"D={rD.get('Random Forest|cons',float('nan')):.3f}")

R = pd.DataFrame(rows)
R.to_csv(OUT / "R3_consensus_per_drug.csv", index=False)
P(f"\n[{time.strftime('%H:%M:%S')}] drugs analysed for consensus: {len(R)}")

def med_ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    rng = np.random.default_rng(SEED)
    bs = [np.median(rng.choice(v, len(v), True)) for _ in range(N_BOOT)]
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

summary = {}
P("\n=== R3 consensus denoising: median AUROC by scheme x target ===")
for mnm in newmodels():
    for tgt in ["cons", "gdsc2", "ctrp"]:
        s = R[f"S_{mnm}|{tgt}"]; c = R[f"C_{mnm}|{tgt}"]; d = R[f"D_{mnm}|{tgt}"]
        both = R[[f"S_{mnm}|{tgt}", f"C_{mnm}|{tgt}", f"D_{mnm}|{tgt}"]].dropna()
        dCS = (both[f"C_{mnm}|{tgt}"] - both[f"S_{mnm}|{tgt}"]).to_numpy()
        dCD = (both[f"C_{mnm}|{tgt}"] - both[f"D_{mnm}|{tgt}"]).to_numpy()
        wilcox = stats.wilcoxon(both[f"C_{mnm}|{tgt}"], both[f"S_{mnm}|{tgt}"]).pvalue if len(both)>5 else np.nan
        mS = np.nanmedian(s); mC = np.nanmedian(c); mD = np.nanmedian(d)
        m_dCS, lo, hi = med_ci(dCS)
        summary[f"{mnm}|{tgt}"] = dict(n=len(both), median_S=round(mS,3), median_C=round(mC,3),
            median_D=round(mD,3), delta_C_minus_S_pts=round(m_dCS*100,2),
            delta_CS_ci=f"{lo*100:.2f}-{hi*100:.2f}", delta_C_minus_Sdown_pts=round(np.median(dCD)*100,2),
            frac_C_gt_S=round(float((dCS>0).mean()),3), wilcoxon_C_vs_S_p=float(wilcox))
        P(f"  {mnm:20s} target={tgt:6s} n={len(both):2d}  S={mS:.3f} C={mC:.3f} D={mD:.3f}  "
          f"ΔC-S={m_dCS*100:+.2f}pts [{lo*100:+.2f},{hi*100:+.2f}]  "
          f"ΔC-Sdown={np.median(dCD)*100:+.2f}pts  frac(C>S)={(dCS>0).mean():.2f}  p={wilcox:.3g}")
(OUT / "R3_consensus_summary.json").write_text(json.dumps(summary, indent=2))
(OUT / "R3_consensus_log.txt").write_text("\n".join(log))
P(f"\n[done] -> {OUT}")
