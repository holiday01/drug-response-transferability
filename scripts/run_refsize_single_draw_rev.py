#!/usr/bin/env python3
"""
REV-1b (revision, 2026-09-30): single-draw reference-size analysis.

The reference-size sweep in run_independent_validation.py averaged REF_DRAWS=20
concordance estimates per drug and split, and analyze_independent_validation.py
then averaged over splits. That measures the mean of many small-reference
estimates, not what a user holding ONE k-line reference set per drug obtains.

Here, for each split s (the saved REV-1 external test set T_s) and each draw d,
every drug gets exactly one random reference subset of k lines drawn from
R_s = (lines with CTRP and GDSC2 fitted AUC) \\ T_s. Concordance on that subset
is compared, across drugs, with the SAME split's external RF AUROC on T_s
(saved in per_drug_split.csv; no models are retrained). Per replicate (s, d):
  - Spearman(concordance, external AUROC) across drugs
  - screening ROC-AUC of concordance for external AUROC >= 0.70
  - agreement of the concordance >= 0.5 call with the full-reference call
The full-reference per-split values are reported alongside for comparison.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from rev_common import load_expr, load_ctrp, load_gdsc2

SEED = 42
from config import REV1_DIR, RESULTS_DIR
D = REV1_DIR
OUT = RESULTS_DIR / "rev1b_refsize_single"
OUT.mkdir(parents=True, exist_ok=True)
REF_SIZES = (20, 50, 100)
DRAWS = 200
THR_AUROC, THR_CONC = 0.70, 0.50
COL = "ext_fitted_Random Forest"

expr = load_expr()
ctrp, name2col = load_ctrp()
pdf = pd.read_csv(D / "per_drug_split.csv")
ids = pd.read_csv(D / "split_ids.csv.gz")
idmap = pd.read_csv(D / "drug_id_map.csv")
drugs = idmap.drug.tolist()
resp, _, _ = load_gdsc2(drugs, lambda *a: None)
g_fit = resp["fitted"]

# reference pools R_s per drug and split (same definition as REV-1)
pools = {}
for r in ids.itertuples():
    c = ctrp[name2col[r.drug]].dropna(); c = c[c.index.isin(expr.index)]
    g = g_fit[r.drug].dropna()
    R = c.index.intersection(g.index).difference(pd.Index(r.test.split(";")))
    pools[(r.drug, r.split)] = (c.loc[R].to_numpy(), g.loc[R].to_numpy())

pdf = pdf.dropna(subset=[COL])
rng = np.random.default_rng(SEED)
rows, full_rows = [], []
for s, x in pdf.groupby("split"):
    x = x.set_index("drug")
    ext = x[COL]; y = (ext >= THR_AUROC).astype(int)
    full_call = x.conc_fitted >= THR_CONC
    full_rows.append(dict(split=s, n_drugs=len(x),
                          spearman=stats.spearmanr(x.conc_fitted, ext).statistic,
                          screen_auc=roc_auc_score(y, x.conc_fitted) if y.nunique() == 2 else np.nan))
    for k in REF_SIZES:
        for d in range(DRAWS):
            conc = {}
            for nm in x.index:
                a, b = pools[(nm, s)]
                if len(a) < k: continue
                i = rng.choice(len(a), size=k, replace=False)
                conc[nm] = stats.spearmanr(a[i], b[i]).statistic
            conc = pd.Series(conc); e = ext.loc[conc.index]; yy = y.loc[conc.index]
            rows.append(dict(k=k, split=s, draw=d, n_drugs=len(conc),
                             spearman=stats.spearmanr(conc, e).statistic,
                             screen_auc=roc_auc_score(yy, conc) if yy.nunique() == 2 else np.nan,
                             call_agreement=float(((conc >= THR_CONC) == full_call.loc[conc.index]).mean())))

rep = pd.DataFrame(rows); rep.to_csv(OUT / "replicates.csv", index=False)
full = pd.DataFrame(full_rows); full.to_csv(OUT / "full_reference_per_split.csv", index=False)


def q(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return dict(median=round(float(np.median(v)), 3), p2_5=round(float(np.percentile(v, 2.5)), 3),
                p97_5=round(float(np.percentile(v, 97.5)), 3), min=round(float(v.min()), 3))


out = dict(seed=SEED, draws_per_split=DRAWS, splits=int(full.split.nunique()),
           unit="one k-line reference per drug, one split; statistics across drugs",
           full_reference_per_split=dict(spearman=q(full.spearman), screen_auc=q(full.screen_auc)))
for k in REF_SIZES:
    r = rep[rep.k == k]
    out[f"ref{k}"] = dict(n_drugs_median=float(r.n_drugs.median()), spearman=q(r.spearman),
                          screen_auc=q(r.screen_auc), call_agreement_with_full=q(r.call_agreement),
                          frac_replicates_spearman_ge_0p7=round(float((r.spearman >= 0.7).mean()), 3))
(OUT / "summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
