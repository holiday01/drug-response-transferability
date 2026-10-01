#!/usr/bin/env python3
"""
REV-2b (revision, 2026-09-30): drug-block permutation for the pooled LOAO test.

run_three_assay_rev.py permuted scores independently within each held-out assay,
which breaks the dependence among a drug's three rows. Here one permutation of
drug labels is applied jointly to all three assays (each drug's three scores
move together), matching the drug-cluster bootstrap. Reads the saved LOAO
tables; no models are retrained. Two-sided, +1 corrected.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED, N_PERM = 42, 5000
import sys; sys.path.insert(0, str(Path(__file__).parent))
from config import REV2_DIR
D = REV2_DIR
out = {"seed": SEED, "permutations": N_PERM, "unit": "drug (3 held-out rows kept together)"}
for mn, f in [("Random Forest", "loo_random.csv"), ("Logistic Regression", "loo_logistic.csv")]:
    d = pd.read_csv(D / f)
    x = d.pivot(index="drug", columns="held_out", values="score")
    y = d.pivot(index="drug", columns="held_out", values="transfer").reindex_like(x)
    assert x.notna().all().all() and y.notna().all().all()
    obs = stats.spearmanr(x.to_numpy().ravel(), y.to_numpy().ravel()).statistic
    rng = np.random.default_rng(SEED); X = x.to_numpy(); Y = y.to_numpy().ravel()
    null = np.array([stats.spearmanr(X[rng.permutation(len(X))].ravel(), Y).statistic for _ in range(N_PERM)])
    out[mn] = dict(n_drugs=len(x), pooled_spearman=round(float(obs), 3),
                   block_permutation_p=float((1 + (np.abs(null) >= abs(obs)).sum()) / (N_PERM + 1)))
(D / "block_permutation.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
