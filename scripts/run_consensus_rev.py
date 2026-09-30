#!/usr/bin/env python3
"""
REV-3 (revision, 2026-09-30): consensus-label training re-run with the official
GDSC2 fitted AUC, repeated splits, train-only label thresholds and information-
matched baselines.

Per drug and split s, on lines with CTRP + GDSC2 (fitted) response + expression:
  train/test = 70/30 random split (seed 42 + s); tertile thresholds for every
  assay are fitted on TRAIN lines only and applied to TEST lines.
Training schemes (features + scaler always fitted on the scheme's own train lines):
  S : CTRP labels, all CTRP-extreme train lines          (source-only)
  G : GDSC2 labels, all GDSC2-extreme train lines        (target-platform-only)
  A : labels from the mean of per-assay rank percentiles (both platforms, averaged)
  C : train lines where CTRP and GDSC2 labels agree      (consensus)
  D : random S subset matched to |C|                     (size control)
  X : the |C| most extreme CTRP train lines, class-balanced (extreme-only control)
Targets on TEST lines:
  cons  : lines where CTRP and GDSC2 agree
  gdsc2 : GDSC2 label
  ctrp  : CTRP label
  prism : PRISM label (third assay, never used for training labels; thresholds
          from PRISM values of non-test lines)
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from rev_common import (load_expr, load_ctrp, load_gdsc2, load_prism, select_features,
                        tert_thresholds, apply_tert)

SEED = 42
SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
from config import RESULTS_DIR, REV3_DIR
OUT = REV3_DIR
OUT.mkdir(parents=True, exist_ok=True)
MIN_SHARED, MIN_AGREE_CLASS, MIN_TEST_CLASS = 120, 12, 8
SCHEMES = ["S", "G", "A", "C", "D", "X"]
TARGETS = ["cons", "gdsc2", "ctrp", "prism"]
N_BOOT = 2000
log_lines = []


def log(m=""):
    print(m, flush=True); log_lines.append(str(m))


log(f"[{time.strftime('%H:%M:%S')}] loading ...")
expr = load_expr()
ctrp, c2col = load_ctrp()
prism = load_prism()
gd_all = pd.read_csv(RESULTS_DIR / "R3_consensus_per_drug.csv").drug.tolist()
resp, id_map, _ = load_gdsc2(sorted(set(gd_all) | set(c2col)), log=lambda m: None)
gfit = resp["fitted"]
drugs = sorted(d for d in gfit if d in c2col)
id_map[id_map.drug.isin(drugs)].assign(ctrp_col=lambda x: x.drug.map(c2col)).to_csv(OUT / "drug_id_map.csv", index=False)
log(f"  candidate drugs with CTRP + fitted GDSC2: {len(drugs)} (original R3 panel: {len(gd_all)})")


def newmodels(seed):
    return {"Random Forest": RandomForestClassifier(n_estimators=400, n_jobs=1, random_state=seed),
            "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=seed)}


def ok(y):
    return len(y) and y.nunique() == 2 and y.value_counts().min() >= MIN_TEST_CLASS


def run_one(nm, s):
    cv = ctrp[c2col[nm]].dropna(); gv = gfit[nm]
    both = cv.index.intersection(gv.index).intersection(expr.index)
    if len(both) < MIN_SHARED:
        return None
    rng = np.random.default_rng(SEED + 1000 * s + drugs.index(nm))
    perm = rng.permutation(both.to_numpy()); nte = int(round(0.3 * len(perm)))
    te, tr = pd.Index(perm[:nte]), pd.Index(perm[nte:])
    cl_tr = apply_tert(cv.loc[tr], *tert_thresholds(cv.loc[tr]))
    gl_tr = apply_tert(gv.loc[tr], *tert_thresholds(gv.loc[tr]))
    cl_te = apply_tert(cv.loc[te], *tert_thresholds(cv.loc[tr]))
    gl_te = apply_tert(gv.loc[te], *tert_thresholds(gv.loc[tr]))
    # averaged label: mean of per-assay percentile ranks among train lines, thresholds from train
    pr = lambda x, ref: x.apply(lambda v: (ref < v).mean() + 0.5 * (ref == v).mean())
    avg_tr = (pr(cv.loc[tr], cv.loc[tr]) + pr(gv.loc[tr], gv.loc[tr])) / 2
    al_tr = apply_tert(avg_tr, *tert_thresholds(avg_tr))

    agree_tr = cl_tr.index.intersection(gl_tr.index)
    C = cl_tr.loc[agree_tr][cl_tr.loc[agree_tr] == gl_tr.loc[agree_tr]]
    if C.value_counts().min() < MIN_AGREE_CLASS or C.nunique() < 2:
        return None
    k = len(C)
    D = cl_tr.loc[rng.choice(cl_tr.index.to_numpy(), size=min(k, len(cl_tr)), replace=False)]
    if D.nunique() < 2 or D.value_counts().min() < 3:
        return None
    # extreme-only: the k/2 lowest and k/2 highest CTRP train values
    srt = cv.loc[tr].sort_values(); h = k // 2
    X = pd.concat([pd.Series(0, index=srt.index[:h]), pd.Series(1, index=srt.index[-h:])])
    train_sets = dict(S=cl_tr, G=gl_tr, A=al_tr, C=C, D=D, X=X)

    agree_te = cl_te.index.intersection(gl_te.index)
    cons_te = cl_te.loc[agree_te][cl_te.loc[agree_te] == gl_te.loc[agree_te]]
    if not ok(cons_te):
        return None
    tgt = dict(cons=cons_te, gdsc2=gl_te, ctrp=cl_te)
    if nm in prism.columns:
        pv = prism[nm].dropna(); pv = pv[pv.index.isin(expr.index)]
        pte = pv[pv.index.isin(te)]
        if len(pte) >= 30:
            tgt["prism"] = apply_tert(pte, *tert_thresholds(pv[~pv.index.isin(te)]))

    row = dict(drug=nm, split=s, n_shared=len(both), n_train_C=k, n_test_cons=len(cons_te),
               n_test_prism=len(tgt.get("prism", [])))
    for sch, y in train_sets.items():
        assert not (set(y.index) & set(te))
        Xtr = expr.loc[y.index].to_numpy(); ytr = y.to_numpy()
        feat = select_features(Xtr, ytr); sc = StandardScaler().fit(Xtr[:, feat])
        Xs = sc.transform(Xtr[:, feat])
        for mn, clf in newmodels(SEED + s).items():
            clf.fit(Xs, ytr)
            for t, yt in tgt.items():
                if not ok(yt):
                    continue
                p = clf.predict_proba(sc.transform(expr.loc[yt.index].to_numpy()[:, feat]))[:, 1]
                row[f"{sch}|{mn}|{t}"] = roc_auc_score(yt.to_numpy(), p)
    return row


log(f"[{time.strftime('%H:%M:%S')}] running {len(drugs)} drugs x {SPLITS} splits ...")
res = Parallel(n_jobs=20, verbose=5)(delayed(run_one)(nm, s) for s in range(SPLITS) for nm in drugs)
df = pd.DataFrame([r for r in res if r is not None])
df.to_csv(OUT / "per_drug_split.csv", index=False)
log(f"  rows {len(df)}, drugs {df.drug.nunique()}")

# ---------------- drug-level summary (mean over splits), paired contrasts vs C
rng = np.random.default_rng(SEED)
agg = df.groupby("drug").mean(numeric_only=True)
summ = {}
for mn in ["Logistic Regression", "Random Forest"]:
    for t in TARGETS:
        cc = f"C|{mn}|{t}"
        if cc not in agg:
            continue
        ent = {"median_auroc": {sch: round(float(agg[f"{sch}|{mn}|{t}"].median()), 4)
                                for sch in SCHEMES if f"{sch}|{mn}|{t}" in agg}}
        for sch in [x for x in SCHEMES if x != "C"]:
            pair = agg[[cc, f"{sch}|{mn}|{t}"]].dropna()
            d = (pair[cc] - pair[f"{sch}|{mn}|{t}"]).to_numpy()
            bs = [np.median(rng.choice(d, len(d))) for _ in range(N_BOOT)]
            ent[f"C_minus_{sch}"] = dict(
                n_drugs=len(d), median_pts=round(float(np.median(d)) * 100, 2),
                ci_pts=[round(float(np.percentile(bs, 2.5)) * 100, 2), round(float(np.percentile(bs, 97.5)) * 100, 2)],
                n_drugs_C_worse=int((d < 0).sum()),
                wilcoxon_p=float(stats.wilcoxon(d).pvalue) if len(d) > 5 else None)
        summ[f"{mn}|{t}"] = ent
(OUT / "summary.json").write_text(json.dumps(summ, indent=2))
(OUT / "run_log.txt").write_text("\n".join(log_lines))
print(json.dumps(summ, indent=2))
