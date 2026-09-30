#!/usr/bin/env python3
"""
REV-1 (revision, 2026-09-30): independent reference/test validation of
concordance -> cross-assay transferability, under three GDSC2 response summaries.

Fixes two issues raised in the pre-submission review:
  (1) In run_external_validation.py the CTRP-GDSC2 concordance was computed on
      ALL shared cell lines, including a median 61.7% of each drug's external
      GDSC2 test lines. Here every split first draws an external TEST set T of
      GDSC2 cell lines; T's labels are never used for concordance, for label
      thresholds, or for training.
  (2) The legacy GDSC2 summary averaged every dose/replicate column of every
      GDSC2 compound sharing a name. Here three summaries are compared:
        fitted : official GDSC2 release 8.5 fitted-curve AUC (primary)
        trapz  : harmonised per-curve AUC from replicate-level data, restricted
                 to each compound's modal dose set (replicates averaged per dose)
        legacy : the original mean-over-all-columns summary (for comparison)

Per drug and split s (SPLITS seeds):
  G  = cell lines with expression and a primary (fitted) GDSC2 response
  T  = random TEST_FRAC of G                          (external test set)
  CTRP training/internal pool = CTRP-labelled lines not in T
      -> 70/30 stratified split into CTRP-train / CTRP-internal-test
  Reference R = lines with CTRP and GDSC2 response, not in T
      -> concordance = Spearman(CTRP, GDSC2) on R, per GDSC2 summary
      -> reference-size sweep: concordance on random subsets of R (REF_SIZES)
  GDSC2 tertile thresholds are fitted on G \\ T and applied to T.
Models are trained once per split and scored on T under each summary.
Everything is seeded; all per-split IDs (test T, reference R under the fitted
summary, CTRP train and internal test) are written to split_ids.csv.gz.
"""
from __future__ import annotations
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import (SUMMARIES, load_expr, load_ctrp, load_gdsc2, select_features,
                        tert_thresholds, apply_tert)

SEED = 42
from config import REPO_DIR, RESULTS_DIR, REV1_DIR
PANEL = str(RESULTS_DIR / "per_drug_external.csv")  # written by run_external_validation.py
OUT = REV1_DIR
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TEST_FRAC = 0.35
MIN_EXT_CLASS = 15
REF_SIZES = (20, 50, 100)
REF_DRAWS = 20
N_JOBS = 20
log_lines: list[str] = []


def log(m=""):
    print(m, flush=True); log_lines.append(str(m))


log(f"[{time.strftime('%H:%M:%S')}] loading data ...")
expr = load_expr()
ctrp, ctrp_name2col = load_ctrp()
panel = pd.read_csv(PANEL)
panel = panel[panel["ext_Random Forest"].notna()].drug.tolist()
log(f"  drug panel (original drugs with an external set): {len(panel)}")
resp, id_map, agree = load_gdsc2(panel, log)
id_map["ctrp_col"] = id_map.drug.map(ctrp_name2col)
id_map.to_csv(OUT / "drug_id_map.csv", index=False)
agree.to_csv(OUT / "summary_agreement.csv", index=False)
drugs = id_map.drug.tolist()
log(f"  median Spearman fitted~trapz={agree.fitted_vs_trapz.median():.3f}, "
    f"fitted~legacy={agree.fitted_vs_legacy.median():.3f} (min {agree.fitted_vs_legacy.min():.3f})")


def models(seed):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=400, n_jobs=1, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", n_jobs=1, random_state=seed),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=seed),
        "DNN (MLP)": MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, max_iter=300,
                                   early_stopping=True, random_state=seed),
    }


def spear(a, b):
    return float(stats.spearmanr(a, b).statistic) if len(a) >= 10 else np.nan


# ------------------------------------------------------------------ one drug x split
def run_one(nm, s):
    rng = np.random.default_rng(SEED + 1000 * s + drugs.index(nm))
    c_all = ctrp[ctrp_name2col[nm]].dropna(); c_all = c_all[c_all.index.isin(expr.index)]
    g_prim = resp["fitted"][nm].dropna(); g_prim = g_prim[g_prim.index.isin(expr.index)]
    G = g_prim.index.to_numpy()
    T = set(rng.choice(G, size=int(round(TEST_FRAC * len(G))), replace=False))

    # CTRP training / internal test, thresholds from non-T lines only
    c_pool = c_all[~c_all.index.isin(T)]
    lo, hi = tert_thresholds(c_pool); yc = apply_tert(c_pool, lo, hi)
    if min(np.bincount(yc.to_numpy(), minlength=2)) < 20:
        return None, None
    tr_ids, in_ids = train_test_split(yc.index.to_numpy(), test_size=0.30,
                                      stratify=yc.to_numpy(), random_state=SEED + s)
    Xtr = expr.loc[tr_ids].to_numpy(); ytr = yc.loc[tr_ids].to_numpy()
    feat = select_features(Xtr, ytr)
    sc = StandardScaler().fit(Xtr[:, feat])
    Xtr_s = sc.transform(Xtr[:, feat])
    Xin_s = sc.transform(expr.loc[in_ids].to_numpy()[:, feat]); yin = yc.loc[in_ids].to_numpy()
    assert not (set(tr_ids) & T) and not (set(in_ids) & T)

    # external labels per summary: thresholds on G\T, applied to T
    ext = {}
    for sm in SUMMARIES:
        g = resp[sm][nm].dropna(); g = g[g.index.isin(expr.index)]
        lo_g, hi_g = tert_thresholds(g[~g.index.isin(T)])
        yT = apply_tert(g[g.index.isin(T)], lo_g, hi_g)
        if yT.nunique() == 2 and min(np.bincount(yT.to_numpy())) >= MIN_EXT_CLASS:
            ext[sm] = yT

    # concordance on reference R (both assays, not in T), per summary
    conc, refs = {}, {}
    for sm in SUMMARIES:
        g = resp[sm][nm].dropna()
        R = c_all.index.intersection(g.index).difference(pd.Index(list(T)))
        assert not (set(R) & T)
        conc[sm] = spear(c_all.loc[R], g.loc[R]); refs[sm] = len(R)
        if sm == "fitted":
            R_fit = R; Rarr = R.to_numpy()
            for k in REF_SIZES:
                vals = []
                for _ in range(REF_DRAWS):
                    if len(Rarr) < k: break
                    sub = rng.choice(Rarr, size=k, replace=False)
                    vals.append(spear(c_all.loc[sub], g.loc[sub]))
                conc[f"fitted_ref{k}"] = float(np.mean(vals)) if vals else np.nan
                conc[f"fitted_ref{k}_sd"] = float(np.std(vals)) if vals else np.nan
    # leaky comparison: concordance on ALL shared lines incl. T (original design)
    gf = resp["fitted"][nm].dropna(); both = c_all.index.intersection(gf.index)
    conc["fitted_leaky_allshared"] = spear(c_all.loc[both], gf.loc[both])

    row = dict(drug=nm, split=s, n_T=len(T), n_train=len(tr_ids), n_internal=len(in_ids),
               **{f"n_ref_{k}": v for k, v in refs.items()},
               **{f"conc_{k}": v for k, v in conc.items()},
               **{f"n_ext_{sm}": len(ext[sm]) if sm in ext else 0 for sm in SUMMARIES})
    for mn, clf in models(SEED + s).items():
        clf.fit(Xtr_s, ytr)
        row[f"int_{mn}"] = float(roc_auc_score(yin, clf.predict_proba(Xin_s)[:, 1]))
        for sm, yT in ext.items():
            XT = sc.transform(expr.loc[yT.index].to_numpy()[:, feat])
            row[f"ext_{sm}_{mn}"] = float(roc_auc_score(yT.to_numpy(), clf.predict_proba(XT)[:, 1]))
    ids = dict(drug=nm, split=s, test=";".join(sorted(T)), reference=";".join(sorted(R_fit)),
               train=";".join(sorted(tr_ids)),
               internal=";".join(sorted(in_ids)))
    return row, ids


log(f"\n[{time.strftime('%H:%M:%S')}] running {len(drugs)} drugs x {SPLITS} splits ...")
jobs = [(nm, s) for s in range(SPLITS) for nm in drugs]
res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_one)(nm, s) for nm, s in jobs)
rows = [r for r, _ in res if r is not None]
idrows = [i for _, i in res if i is not None]
df = pd.DataFrame(rows)
df.to_csv(OUT / "per_drug_split.csv", index=False)
pd.DataFrame(idrows).to_csv(OUT / "split_ids.csv.gz", index=False, compression="gzip")
log(f"  rows: {len(df)}")
log(f"[{time.strftime('%H:%M:%S')}] done. -> {OUT.relative_to(REPO_DIR) if OUT.is_relative_to(REPO_DIR) else OUT}")
(OUT / "run_log.txt").write_text("\n".join(log_lines))
json.dump(dict(seed=SEED, splits=SPLITS, test_frac=TEST_FRAC, top_k=200, var_q=0.50,
               min_ext_class=MIN_EXT_CLASS, ref_sizes=REF_SIZES, ref_draws=REF_DRAWS,
               fitted_file="GDSC2_fitted_dose_response_27Oct23.xlsx", generated=time.strftime("%Y-%m-%d %H:%M:%S")),
          open(OUT / "run_meta.json", "w"), indent=2)
