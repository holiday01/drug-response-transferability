#!/usr/bin/env python3
"""
REV-2 (revision, 2026-09-30): leave-one-assay-out (LOAO) re-run with the official
GDSC2 fitted AUC, repeated splits, and drug-clustered inference.

For every ordered pair (train A -> test B) and split s, a model trained on A
(70% of A's tertile-labelled lines; thresholds from A's non-test lines) is scored
on B's tertile-labelled lines not used in training. For held-out assay H the
drug's score is the concordance of the two OTHER assays (never touches H's
labels); the outcome is mean external AUROC of transfers INTO H.

Inference: each drug contributes up to three rows (one per held-out assay), so
the pooled correlation is reported with a drug-cluster bootstrap and a
within-assay drug permutation null; per-held-out-assay correlations (one row per
drug) are the primary read-out.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import (load_expr, load_ctrp, load_gdsc2, load_prism, select_features,
                        tert_thresholds, apply_tert)

SEED = 42
SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
from config import REV2_DIR
OUT = REV2_DIR
OUT.mkdir(parents=True, exist_ok=True)
MIN_LAB, MIN_CLASS, N_BOOT, N_PERM = 120, 15, 5000, 5000
ASSAYS = ["CTRP", "GDSC2", "PRISM"]
log_lines = []


def log(m=""):
    print(m, flush=True); log_lines.append(str(m))


log(f"[{time.strftime('%H:%M:%S')}] loading ...")
expr = load_expr()
ctrp, c2col = load_ctrp()
prism = load_prism()
cand = sorted(set(c2col) & set(prism.columns))
resp, id_map, _ = load_gdsc2(cand, log=lambda m: None)
A = {"CTRP": {nm: ctrp[c2col[nm]].dropna() for nm in cand},
     "GDSC2": resp["fitted"],
     "PRISM": {nm: prism[nm].dropna() for nm in cand}}
drugs = sorted(set(A["GDSC2"]) & set(cand))
id_map[id_map.drug.isin(drugs)].assign(ctrp_col=lambda x: x.drug.map(c2col)).to_csv(OUT / "drug_id_map.csv", index=False)
log(f"  drugs measured in all three assays (name-matched): {len(drugs)}")


def run_pair(nm, src, dst, s):
    a = A[src][nm]; a = a[a.index.isin(expr.index)]
    b = A[dst][nm]; b = b[b.index.isin(expr.index)]
    ya = apply_tert(a, *tert_thresholds(a))
    if len(ya) < MIN_LAB or ya.value_counts().min() < 20:
        return None
    tr, te = train_test_split(ya.index.to_numpy(), test_size=0.3, stratify=ya.to_numpy(), random_state=SEED + s)
    ytr = apply_tert(a.loc[tr], *tert_thresholds(a[~a.index.isin(te)]))  # thresholds without test lines
    b_ext = b[~b.index.isin(ytr.index)]
    yb = apply_tert(b_ext, *tert_thresholds(b_ext))
    if yb.nunique() < 2 or yb.value_counts().min() < MIN_CLASS:
        return None
    Xtr = expr.loc[ytr.index].to_numpy(); feat = select_features(Xtr, ytr.to_numpy())
    sc = StandardScaler().fit(Xtr[:, feat]); Xs = sc.transform(Xtr[:, feat])
    Xb = sc.transform(expr.loc[yb.index].to_numpy()[:, feat])
    both = A[src][nm].index.intersection(A[dst][nm].index)
    r = dict(drug=nm, src=src, dst=dst, split=s, n_ext=len(yb),
             rho=float(stats.spearmanr(A[src][nm].loc[both], A[dst][nm].loc[both]).statistic) if len(both) >= 20 else np.nan)
    for mn, clf in {"Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=SEED + s),
                    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED + s)}.items():
        clf.fit(Xs, ytr.to_numpy())
        r[f"ext_{mn}"] = float(roc_auc_score(yb.to_numpy(), clf.predict_proba(Xb)[:, 1]))
    return r


pairs = [(x, y) for x in ASSAYS for y in ASSAYS if x != y]
res = Parallel(n_jobs=20, verbose=5)(delayed(run_pair)(nm, a, b, s)
                                     for s in range(SPLITS) for nm in drugs for a, b in pairs)
df = pd.DataFrame([r for r in res if r is not None])
df.to_csv(OUT / "per_pair_split.csv", index=False)

rng = np.random.default_rng(SEED)
summary = {"n_drugs_all_three": len(drugs)}
for mn in ["Random Forest", "Logistic Regression"]:
    pp = df.groupby(["drug", "src", "dst"]).agg(rho=("rho", "first"), ext=(f"ext_{mn}", "mean")).reset_index()
    loo = []
    for nm, d in pp.groupby("drug"):
        for H in ASSAYS:
            into = d[d.dst == H].ext.dropna()
            other = d[(d.src != H) & (d.dst != H)].rho.dropna()
            if len(into) and len(other):
                loo.append(dict(drug=nm, held_out=H, score=float(other.mean()), transfer=float(into.mean())))
    loo = pd.DataFrame(loo)
    # restrict to drugs with all three held-out rows (balanced design, as in the original)
    full = loo.groupby("drug").held_out.nunique(); loo = loo[loo.drug.isin(full[full == 3].index)]
    loo.to_csv(OUT / f"loo_{mn.split()[0].lower()}.csv", index=False)
    ent = {"n_drugs": int(loo.drug.nunique()), "n_rows": len(loo)}
    for H in ASSAYS:
        x = loo[loo.held_out == H]; sp = stats.spearmanr(x.score, x.transfer)
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(x), len(x)); bs.append(stats.spearmanr(x.score.values[i], x.transfer.values[i]).statistic)
        ent[H] = dict(n=len(x), spearman=round(float(sp.statistic), 3), p=float(sp.pvalue),
                      ci=[round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)])
    obs = stats.spearmanr(loo.score, loo.transfer).statistic
    ds = loo.drug.unique(); bs = []
    for _ in range(N_BOOT):
        pick = rng.choice(ds, len(ds)); x = pd.concat([loo[loo.drug == d] for d in pick])
        bs.append(stats.spearmanr(x.score, x.transfer).statistic)
    null = []
    for _ in range(N_PERM):
        x = loo.copy()
        x["score"] = x.groupby("held_out").score.transform(lambda v: rng.permutation(v.values))
        null.append(stats.spearmanr(x.score, x.transfer).statistic)
    ent["pooled"] = dict(spearman=round(float(obs), 3),
                         drug_cluster_bootstrap_ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                         within_assay_permutation_p=float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)))
    summary[mn] = ent
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
(OUT / "run_log.txt").write_text("\n".join(log_lines))
print(json.dumps(summary, indent=2))
