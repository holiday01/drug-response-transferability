#!/usr/bin/env python3
"""REV-3b (revision, 2026-09-30): valid-split counts and minimum-split sensitivity
for the label-combination analysis. Ten splits were attempted per drug; a split is
valid when it met the class-size requirements. Recomputes the median consensus-minus-
comparator differences on all drugs and on drugs with all ten valid splits.
Reads per_drug_split.csv; no models are retrained."""
import json
import pandas as pd
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).parent))
from config import REV3_DIR
D = REV3_DIR
df = pd.read_csv(D / "per_drug_split.csv")
nv = df.dropna(subset=["S|Logistic Regression|gdsc2"]).groupby("drug").size()
nvp = df.dropna(subset=["C|Logistic Regression|prism"]).groupby("drug").size()
out = {}
for keep, lab in [(nv.index, "all"), (nv[nv == 10].index, "ten_valid_splits")]:
    agg = df[df.drug.isin(keep)].groupby("drug").mean(numeric_only=True)
    for mn in ["Logistic Regression", "Random Forest"]:
        for t in ["gdsc2", "prism"]:
            cc = f"C|{mn}|{t}"
            for sch in "SGA":
                p = agg[[cc, f"{sch}|{mn}|{t}"]].dropna(); d = (p[cc] - p[f"{sch}|{mn}|{t}"]) * 100
                out[f"{lab}|{mn[:3]}|{t}|C-{sch}"] = [len(d), round(float(d.median()), 2)]
out["valid_splits_gdsc2"] = {k: int(v) for k, v in nv.items()}
out["valid_splits_prism"] = {k: int(v) for k, v in nvp.items()}
(D / "min_splits_sensitivity.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    if "|" in k: print(k, v)
