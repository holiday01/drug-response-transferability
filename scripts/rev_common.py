"""Shared loaders for the 2026-09-30 revision analyses (REV-1, REV-3).

GDSC2 response summaries (low = sensitive for all three):
  fitted : official GDSC2 release 8.5 fitted-curve AUC (primary)
  trapz  : harmonised per-curve AUC from replicate-level data over the compound's
           modal dose set (replicates averaged per dose)
  legacy : original mean over every dose/replicate column sharing the drug name
For each drug name the GDSC2 compound ID with the most fitted cell lines is used.
"""
from __future__ import annotations
import json, re
import numpy as np
import pandas as pd
from config import CCLE_DIR, PRISM_AUC, GDSC2_FITTED

BASE = str(CCLE_DIR)
EXPR = f"{BASE}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
CTRP = f"{BASE}/Drug_sensitivity_AUC_(CTD^2)_subsetted.csv"
GDSC = f"{BASE}/Drug_sensitivity_replicate-level_dose_(Sanger_GDSC2)_subsetted.csv"
MODEL = f"{BASE}/Model.csv"
PRISM = str(PRISM_AUC)
FITTED = str(GDSC2_FITTED)
SUMMARIES = ("fitted", "trapz", "legacy")
COLRE = re.compile(r"^(.*?)\s*\(GDSC2:(\d+)\)\s+([0-9.eE+-]+)μM\s+rep(\d+)$")


def base_name(col, portal):
    return re.split(rf"\s*\({portal}:", col)[0].strip().upper()


def load_expr():
    expr = pd.read_csv(EXPR)
    if "IsDefaultEntryForModel" in expr.columns:
        expr = expr[expr["IsDefaultEntryForModel"].astype(str).str.strip() == "Yes"]
    meta = {"", "SequencingID", "ModelID", "IsDefaultEntryForModel", "ModelConditionID", "IsDefaultEntryForMC"}
    mc = [c for c in expr.columns if c in meta or str(c).lower().startswith("unnamed")]
    return (expr.dropna(subset=["ModelID"]).drop_duplicates("ModelID").set_index("ModelID")
                .drop(columns=[c for c in mc if c != "ModelID"], errors="ignore").astype(np.float32))


def load_ctrp():
    """Returns (matrix, name -> best-covered column)."""
    ctrp = pd.read_csv(CTRP, index_col=0)
    ctrp = ctrp[[c for c in ctrp.columns if ":" not in c.split("(")[0]]]
    name2col = {}
    for c in ctrp.columns:
        nm = base_name(c, "CTRP")
        if nm not in name2col or ctrp[c].notna().sum() > ctrp[name2col[nm]].notna().sum():
            name2col[nm] = c
    return ctrp, name2col


def load_prism():
    d = pd.read_csv(PRISM, usecols=["depmap_id", "name", "auc"]).dropna()
    d["name"] = d["name"].astype(str).str.upper().str.strip()
    return d.groupby(["depmap_id", "name"])["auc"].mean().unstack("name")


def load_gdsc2(drugs, log=print):
    """Returns ({summary: {drug: Series}}, id-map DataFrame, agreement DataFrame)."""
    from scipy import stats
    gdsc = pd.read_csv(GDSC, index_col=0); gdsc.index = gdsc.index.astype(str)
    info = [(c, m.group(1).strip().upper(), int(m.group(2)), float(m.group(3)), int(m.group(4)))
            for c in gdsc.columns if (m := COLRE.match(c))]
    info = pd.DataFrame(info, columns=["col", "name", "drug_id", "dose", "rep"])
    log(f"  GDSC2 replicate columns parsed: {len(info)}/{gdsc.shape[1]}")

    def trapz_auc(g):
        doses = np.sort(g.dose.unique())
        per_dose = pd.DataFrame({d: gdsc[g[g.dose == d].col].mean(axis=1, skipna=True) for d in doses})
        cover = per_dose.notna().mean()
        use = np.sort(cover[cover >= 0.8 * cover.max()].index.to_numpy())
        if len(use) < 4:
            return pd.Series(np.nan, index=gdsc.index)
        v = per_dose[use]; ok = v.notna().all(axis=1); x = np.log10(use)
        auc = np.trapezoid(v[ok].to_numpy(), x, axis=1) / (x[-1] - x[0])
        return pd.Series(auc, index=v.index[ok]).reindex(gdsc.index)

    fit = pd.read_excel(FITTED)
    mdl = pd.read_csv(MODEL, usecols=["ModelID", "SangerModelID"])
    sid2mid = mdl.dropna().drop_duplicates("SangerModelID").set_index("SangerModelID").ModelID
    fit["ModelID"] = fit["SANGER_MODEL_ID"].map(sid2mid)
    log(f"  fitted rows {len(fit)}; mapped to ModelID {fit.ModelID.notna().mean():.3f}")
    fit_piv = fit.dropna(subset=["ModelID"]).pivot_table(index="ModelID", columns="DRUG_ID",
                                                         values="AUC", aggfunc="mean")
    resp = {s: {} for s in SUMMARIES}; id_map = []
    for nm in drugs:
        ids = sorted(info[info.name == nm].drug_id.unique())
        ids_fit = [i for i in ids if i in fit_piv.columns]
        if not ids_fit:
            log(f"  !! {nm}: no fitted AUC for GDSC2 IDs {ids}"); continue
        best = max(ids_fit, key=lambda i: fit_piv[i].notna().sum())
        resp["fitted"][nm] = fit_piv[best].dropna()
        resp["trapz"][nm] = trapz_auc(info[(info.name == nm) & (info.drug_id == best)]).dropna()
        resp["legacy"][nm] = gdsc[info[info.name == nm].col].mean(axis=1, skipna=True).dropna()
        id_map.append(dict(drug=nm, gdsc2_ids=";".join(map(str, ids)), chosen_id=best,
                           n_cols_per_id=json.dumps({int(k): int(v) for k, v in
                                                     info[info.name == nm].groupby("drug_id").size().items()}),
                           n_fitted=int(fit_piv[best].notna().sum())))
    agree = []
    for d in id_map:
        nm = d["drug"]; a = pd.concat({s: resp[s][nm] for s in SUMMARIES}, axis=1).dropna()
        agree.append(dict(drug=nm, n=len(a), fitted_vs_trapz=stats.spearmanr(a.fitted, a.trapz).statistic,
                          fitted_vs_legacy=stats.spearmanr(a.fitted, a.legacy).statistic))
    return resp, pd.DataFrame(id_map), pd.DataFrame(agree)


def select_features(Xtr, ytr, k=200, var_q=0.50):
    """Variance pre-filter then |point-biserial| univariate ranking, TRAIN ONLY."""
    v = Xtr.var(axis=0); keep = np.where(v >= np.quantile(v, var_q))[0]
    Xk = Xtr[:, keep]; y = ytr.astype(float); n1 = y.sum(); n0 = len(y) - n1
    if n1 < 2 or n0 < 2:
        return keep[:k]
    m1 = Xk[y == 1].mean(0); m0 = Xk[y == 0].mean(0); sd = Xk.std(0); sd[sd == 0] = np.nan
    p = n1 / len(y); r = np.nan_to_num(np.abs((m1 - m0) / sd) * np.sqrt(p * (1 - p)))
    return keep[np.argsort(r)[::-1][:k]]


def tert_thresholds(s):
    s = s.dropna(); return s.quantile(1 / 3), s.quantile(2 / 3)


def apply_tert(s, lo, hi):
    """1 = resistant (>= hi), 0 = sensitive (<= lo); middle dropped."""
    s = s.dropna(); s = s[(s <= lo) | (s >= hi)]
    return (s >= hi).astype(int)
