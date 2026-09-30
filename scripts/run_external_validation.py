#!/usr/bin/env python3
"""
External-validation arm for the reproducibility benchmark.

Adds the missing axis to the controlled benchmark: cross-dataset (cross-assay)
generalization. A model trained under the SAME leakage-free protocol on CCLE
expression -> CTRP sensitivity is evaluated on two test sets drawn from the
identical fitted model:

  (a) INTERNAL : held-out CTRP cell lines (same assay, same distribution)
  (b) EXTERNAL : GDSC2 cell lines (independent assay/lab), labels derived from
                 GDSC2 replicate-level dose-response, with any cell line seen
                 in CTRP training removed -> genuinely unseen + independent.

The internal-minus-external drop isolates the reproducibility cost of the
*validation-design* choice most consequential in practice -- whether a model
is validated on an independent dataset at all -- separate from algorithm and
preprocessing. AUROC is rank-based, so the GDSC2 metric only needs to order
sensitivity consistently (verified per drug by Spearman concordance with CTRP).

Data (real, local, read-only; DepMap 25Q3, all ModelID-indexed):
  X       : CCLE TPM-log1p protein-coding expression
  y_ctrp  : CTRP/CTD^2 drug-sensitivity AUC (low = sensitive)
  y_gdsc2 : mean viability across GDSC2 replicate-level doses (low = sensitive)

All randomness uses SEED. Every reported value is regenerated from these files.
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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
BASE = str(CCLE_DIR)
EXPR = f"{BASE}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
CTRP = f"{BASE}/Drug_sensitivity_AUC_(CTD^2)_subsetted.csv"
GDSC = f"{BASE}/Drug_sensitivity_replicate-level_dose_(Sanger_GDSC2)_subsetted.csv"
OUT  = Path(str(RESULTS_DIR))
OUT.mkdir(parents=True, exist_ok=True)

MIN_LABELLED   = 200   # CTRP labelled cell lines required per drug (match main)
TOP_K          = 200   # univariate features kept (train-derived), match main
VAR_Q          = 0.50  # variance pre-filter quantile, match main P1
MIN_EXT_CLASS  = 15    # min GDSC2 external cell lines per class for a stable AUROC
N_BOOT         = 2000  # bootstrap resamples for CI over drugs
log_lines: list[str] = []


def log(m: str) -> None:
    print(m, flush=True)
    log_lines.append(m)


# ---------------------------------------------------------------- CCLE expression
log(f"[{time.strftime('%H:%M:%S')}] loading CCLE expression ...")
expr = pd.read_csv(EXPR)
if "IsDefaultEntryForModel" in expr.columns:
    expr = expr[expr["IsDefaultEntryForModel"].astype(str).str.strip() == "Yes"]
META = {"", "SequencingID", "ModelID", "IsDefaultEntryForModel",
        "ModelConditionID", "IsDefaultEntryForMC"}
meta_cols = [c for c in expr.columns if c in META or str(c).lower().startswith("unnamed")]
expr = (expr.dropna(subset=["ModelID"]).drop_duplicates("ModelID")
            .set_index("ModelID")
            .drop(columns=[c for c in meta_cols if c != "ModelID"], errors="ignore"))
expr = expr.astype(np.float32)
log(f"  expression: {expr.shape[0]} cell lines x {expr.shape[1]} genes")

# ---------------------------------------------------------------- CTRP AUC
ctrp = pd.read_csv(CTRP, index_col=0)
mono = [c for c in ctrp.columns if ":" not in c.split("(")[0]]
ctrp = ctrp[mono]


def base_name(col: str, portal: str) -> str:
    return re.split(rf"\s*\({portal}:", col)[0].strip().upper()


ctrp_name2col: dict[str, str] = {}
for c in ctrp.columns:
    nm = base_name(c, "CTRP")
    # if a drug name maps to several CTRP columns, keep the best-covered one
    if nm not in ctrp_name2col or ctrp[c].notna().sum() > ctrp[ctrp_name2col[nm]].notna().sum():
        ctrp_name2col[nm] = c

# ---------------------------------------------------------------- GDSC2 -> per-drug mean viability
log(f"[{time.strftime('%H:%M:%S')}] loading GDSC2 replicate-level dose ...")
gdsc = pd.read_csv(GDSC, index_col=0)
gdsc.index = gdsc.index.astype(str)
log(f"  GDSC2: {gdsc.shape[0]} cell lines x {gdsc.shape[1]} dose/replicate columns")

# group GDSC2 dose/replicate columns by drug name, average -> sensitivity metric
gd_groups: dict[str, list[str]] = {}
for c in gdsc.columns:
    gd_groups.setdefault(base_name(c, "GDSC2"), []).append(c)
gdsc2_metric = pd.DataFrame(
    {nm: gdsc[cols].mean(axis=1, skipna=True) for nm, cols in gd_groups.items()},
    index=gdsc.index,
)
log(f"  GDSC2 drugs summarised (mean viability over doses): {gdsc2_metric.shape[1]}")

# sanity: viability scale (lower = more drug-killed = more sensitive)
_v = gdsc2_metric.to_numpy(dtype=float)
_v = _v[np.isfinite(_v)]
log(f"  GDSC2 metric range approx [{np.nanpercentile(_v,1):.3g}, {np.nanpercentile(_v,99):.3g}]")

# ---------------------------------------------------------------- shared drugs
shared_names = sorted(set(ctrp_name2col) & set(gdsc2_metric.columns))
log(f"[{time.strftime('%H:%M:%S')}] CTRP∩GDSC2 shared drug names: {len(shared_names)}")

GENE_COLS = expr.columns.to_numpy()


def tertile_binary(s: pd.Series):
    """1 = resistant (high value), 0 = sensitive (low value); middle dropped."""
    s = s.dropna()
    lo, hi = s.quantile(1 / 3), s.quantile(2 / 3)
    sens, res = s[s <= lo].index, s[s >= hi].index
    y = pd.Series(0, index=sens.append(res)); y.loc[res] = 1
    return y


def select_features(Xtr, ytr, k=TOP_K, var_q=VAR_Q):
    """Variance pre-filter then |point-biserial| univariate selection, TRAIN ONLY.
    Vectorised point-biserial (identical to per-gene scipy) for speed."""
    v = Xtr.var(axis=0)
    keep = np.where(v >= np.quantile(v, var_q))[0]
    Xk = Xtr[:, keep]
    y = ytr.astype(float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 < 2 or n0 < 2:
        return keep[:k]
    m1 = Xk[y == 1].mean(axis=0); m0 = Xk[y == 0].mean(axis=0)
    sd = Xk.std(axis=0); sd[sd == 0] = np.nan
    p = n1 / len(y)
    r = np.abs((m1 - m0) / sd) * np.sqrt(p * (1 - p))
    r = np.nan_to_num(r)
    return keep[np.argsort(r)[::-1][:k]]


def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED),
        "XGBoost":             XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8,
                                             eval_metric="logloss", n_jobs=-1, random_state=SEED),
        "SVM (RBF)":           SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED),
        "DNN (MLP)":           MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3,
                                             max_iter=300, early_stopping=True, random_state=SEED),
    }


def prob(clf, X):
    return clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)


# ---------------------------------------------------------------- per-drug internal vs external
per_drug = []
log(f"\n[{time.strftime('%H:%M:%S')}] === per-drug internal (CTRP held-out) vs external (GDSC2) ===")
for nm in shared_names:
    ccol = ctrp_name2col[nm]
    y_ctrp_all = tertile_binary(ctrp[ccol])
    y_ctrp_all = y_ctrp_all[y_ctrp_all.index.isin(expr.index)]
    if len(y_ctrp_all) < MIN_LABELLED or y_ctrp_all.nunique() < 2:
        continue

    # GDSC2 external labels (independent assay)
    y_gd_all = tertile_binary(gdsc2_metric[nm])
    y_gd_all = y_gd_all[y_gd_all.index.isin(expr.index)]

    # CTRP leakage-free split (identical protocol to main benchmark)
    Xc = expr.loc[y_ctrp_all.index].to_numpy()
    yc = y_ctrp_all.to_numpy()
    if min(np.bincount(yc)) < 20:
        continue
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        Xc, yc, y_ctrp_all.index.to_numpy(), test_size=0.30, stratify=yc, random_state=SEED)

    feat = select_features(Xtr, ytr)
    sc = StandardScaler().fit(Xtr[:, feat])
    Xtr_s = sc.transform(Xtr[:, feat])
    Xte_s = sc.transform(Xte[:, feat])

    # external set: GDSC2 extremes, excluding any cell line used in CTRP training
    ext_idx = y_gd_all.index[~y_gd_all.index.isin(set(idx_tr))]
    y_ext = y_gd_all.loc[ext_idx]
    have_ext = (y_ext.nunique() == 2 and min(np.bincount(y_ext.to_numpy())) >= MIN_EXT_CLASS)

    # concordance of the two independent sensitivity readouts on shared cell lines
    both = ctrp[ccol].dropna().index.intersection(gdsc2_metric[nm].dropna().index)
    rho = np.nan
    if len(both) >= 20:
        rho = stats.spearmanr(ctrp[ccol].loc[both], gdsc2_metric[nm].loc[both]).statistic

    if have_ext:
        Xext_s = sc.transform(expr.loc[ext_idx].to_numpy()[:, feat])
        yext = y_ext.to_numpy()

    row = dict(drug=nm, n_ctrp=int(len(yc)), n_ext=int(len(y_ext)) if have_ext else 0,
               n_ext_pos=int(y_ext.sum()) if have_ext else 0,
               spearman_ctrp_gdsc2=None if np.isnan(rho) else round(float(rho), 3))
    for mn, clf in models().items():
        clf.fit(Xtr_s, ytr)
        row[f"int_{mn}"] = round(float(roc_auc_score(yte, prob(clf, Xte_s))), 4)
        row[f"ext_{mn}"] = round(float(roc_auc_score(yext, prob(clf, Xext_s))), 4) if have_ext else None
    per_drug.append(row)
    tag = "" if have_ext else "  (no external: too few GDSC2 extremes)"
    log(f"  {nm:22s} n_ctrp={row['n_ctrp']:3d} n_ext={row['n_ext']:3d} "
        f"rho={row['spearman_ctrp_gdsc2']}  RF int={row['int_Random Forest']} "
        f"ext={row['ext_Random Forest']}{tag}")

pd_df = pd.DataFrame(per_drug)
pd_df.to_csv(OUT / "per_drug_external.csv", index=False)
ext_df = pd_df[pd_df["ext_Random Forest"].notna()].copy()
log(f"\n  drugs with a valid external set: {len(ext_df)} / {len(pd_df)}")


def boot_ci(vals, n=N_BOOT):
    vals = np.asarray(vals, dtype=float); rng = np.random.default_rng(SEED)
    bs = [np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------------------------------------------------------------- per-model summary table
log(f"\n[{time.strftime('%H:%M:%S')}] === TABLE: internal vs external by model "
    f"(median over {len(ext_df)} shared drugs) ===")
summary = []
for mn in models():
    it = ext_df[f"int_{mn}"].to_numpy(dtype=float)
    et = ext_df[f"ext_{mn}"].to_numpy(dtype=float)
    drop = it - et
    ilo, ihi = boot_ci(it); elo, ehi = boot_ci(et); dlo, dhi = boot_ci(drop)
    r = dict(model=mn,
             median_internal_auroc=round(float(np.median(it)), 3),
             internal_ci=f"{ilo:.3f}-{ihi:.3f}",
             median_external_auroc=round(float(np.median(et)), 3),
             external_ci=f"{elo:.3f}-{ehi:.3f}",
             median_drop_points=round(float(np.median(drop)) * 100, 1),
             drop_ci_points=f"{dlo*100:.1f}-{dhi*100:.1f}",
             n_drugs=int(len(ext_df)))
    log(f"  {mn:22s} internal={r['median_internal_auroc']:.3f}  "
        f"external={r['median_external_auroc']:.3f}  "
        f"drop={r['median_drop_points']:+.1f} pts [{r['drop_ci_points']}]")
    summary.append(r)
sum_df = pd.DataFrame(summary)
sum_df.to_csv(OUT / "table_external_validation.csv", index=False)

# fraction of external AUROCs that fall to near-chance (<0.6) -- reproducibility read
frac_chance = float((ext_df[[f"ext_{m}" for m in models()]].to_numpy(dtype=float) < 0.60).mean())
med_rho = float(np.nanmedian(pd_df["spearman_ctrp_gdsc2"].astype(float)))

meta = dict(seed=SEED, expr=EXPR, ctrp=CTRP, gdsc2=GDSC,
            shared_drug_names=len(shared_names),
            drugs_with_external_set=int(len(ext_df)),
            min_ext_per_class=MIN_EXT_CLASS, top_k=TOP_K, var_q=VAR_Q,
            gdsc2_summary="mean viability across replicate-level doses (low=sensitive)",
            median_spearman_ctrp_gdsc2=round(med_rho, 3),
            frac_external_auroc_below_0p60=round(frac_chance, 3),
            median_external_drop_points={r["model"]: r["median_drop_points"] for r in summary},
            generated=time.strftime("%Y-%m-%d %H:%M:%S"))
(OUT / "external_validation_meta.json").write_text(json.dumps(meta, indent=2))
(OUT / "external_validation_log.txt").write_text("\n".join(log_lines))
log(f"\n[{time.strftime('%H:%M:%S')}] median CTRP-GDSC2 Spearman concordance = {med_rho:.3f}")
log(f"[{time.strftime('%H:%M:%S')}] done. Outputs -> {OUT}")
