#!/usr/bin/env python3
"""Verify every revised headline number in main.tex against the REV result files.
Each check: (label, value computed from results, string that must appear in main.tex)."""
import json, re
import pandas as pd

from config import RESULTS_DIR, MANUSCRIPT_TEX
R = str(RESULTS_DIR)
# The manuscript source is not distributed. If it is available (DRT_MANUSCRIPT_TEX or
# manuscript/main.tex), each quoted phrase is also checked against it; otherwise only
# the value recomputed from results/ is checked against the phrase quoted below.
HAVE_TEX = MANUSCRIPT_TEX.is_file()
TEXN = re.sub(r"\s+", " ", MANUSCRIPT_TEX.read_text()) if HAVE_TEX else ""
S1 = json.load(open(f"{R}/rev1_independent/summary.json"))
EX = json.load(open(f"{R}/rev1_independent/extra_summary.json"))
S2 = json.load(open(f"{R}/rev2_three_assay/summary.json"))
S3 = json.load(open(f"{R}/rev3_consensus/summary.json"))
cv = S1["concordance_vs_external"]
f2 = lambda x: f"{x:.2f}"
f1 = lambda x: f"{x:.1f}"
f3 = lambda x: f"{x:.3f}"
rf = cv["fitted|Random Forest"]
M = ["Logistic Regression", "Random Forest", "XGBoost", "SVM (RBF)", "DNN (MLP)"]
sp = [cv[f"fitted|{m}"]["spearman"] for m in M]
pa = [cv[f"fitted|{m}"]["partial_internal"][0] for m in M]
lr3, rf3 = S3["Logistic Regression|gdsc2"], S3["Random Forest|gdsc2"]
lrp = S3["Logistic Regression|prism"]
checks = [
    ("n drugs", str(S1["n_drugs"]), "58 drugs"),
    ("RF spearman", f2(rf["spearman"]), "Spearman 0.85, 95\\% confidence interval 0.71--0.92"),
    ("RF CI", f"{f2(rf['ci'][0])}--{f2(rf['ci'][1])}", "0.71--0.92"),
    ("5-model range", f"{f2(min(sp))}--{f2(max(sp))}", "0.85--0.88"),
    ("per-split range", f"{f2(rf['per_split_range'][0])} to {f2(rf['per_split_range'][1])}", "0.74 to 0.86 across the ten individual"),
    ("partial RF", f2(rf["partial_internal"][0]), "remained 0.67 for random forest"),
    ("partial P", f"{rf['partial_internal'][1]:.1e}".replace("e-0", "\\times10^{-").rstrip() + "}", "8.9\\times10^{-9}"),
    ("partial range", f"{f2(min(pa))} to {f2(max(pa))}", "0.67 to 0.71 across algorithms"),
    ("OLS R2 range", f"{min(v['ols_r2'] for k, v in EX['table1'].items() if k != 'Model average')*100:.0f}\\% to {max(v['ols_r2'] for k, v in EX['table1'].items() if k != 'Model average')*100:.0f}\\%", "70\\% to 76\\%"),
    ("leaky", f2(S1["leaky_vs_independent_RF_fitted"]["spearman_leaky"]), "gave a Spearman correlation of 0.85"),
    ("conc diff", f3(S1["leaky_vs_independent_RF_fitted"]["median_abs_diff_concordance"]), "median of 0.006"),
    ("trapz range", f"{f2(min(cv[f'trapz|{m}']['spearman'] for m in M))}--{f2(max(cv[f'trapz|{m}']['spearman'] for m in M))}", "0.83--0.85"),
    ("legacy range", f"{f2(min(cv[f'legacy|{m}']['spearman'] for m in M))}--{f2(max(cv[f'legacy|{m}']['spearman'] for m in M))}", "0.81--0.84"),
    ("summary agreement", f2(S1["gdsc2_summary_agreement"]["median_fitted_vs_legacy"]), "median Spearman 0.90"),
    ("crizotinib", f2(S1["gdsc2_summary_agreement"]["min_fitted_vs_legacy"]), "crizotinib (0.31)"),
    ("ref20", f2(S1["reference_size_sweep_RF_fitted"]["ref20"]["spearman"]), "0.83, 0.85 and 0.84"),
    ("ref50", f2(S1["reference_size_sweep_RF_fitted"]["ref50"]["spearman"]), "0.83, 0.85 and 0.84"),
    ("ref100", f2(S1["reference_size_sweep_RF_fitted"]["ref100"]["spearman"]), "0.83, 0.85 and 0.84"),
    ("ref sd", f2(S1["reference_size_sweep_RF_fitted"]["ref20"]["median_within_drug_sd_of_concordance"]), "0.19 with 20 cell lines, 0.11 with 50 and 0.07"),
    ("median n ref", str(int(S1["reference_size_sweep_RF_fitted"]["full_reference"]["median_n_ref"])), "median 333"),
    ("test lines", str(int(S1["test_lines_median"])), "median 247"),
    ("screen AUC", f2(EX["screen_auc_drug_level"]), "ROC-AUC of 0.91 for random"),
    ("screen per split", f"{f2(S1['screen_RF_fitted']['screen_auc_range'][0])} to {f2(S1['screen_RF_fitted']['screen_auc_range'][1])}", "0.85 to 0.92 across splits"),
    ("model avg screen", f2(EX["screen_auc_model_average"]), "0.90 for the model average"),
    ("n transferable", str(EX["n_transferable"]), "25 of 58 drugs"),
    ("rho>=0.5 n", str(EX["table2"][3]["n"]), "(22 of 58)"),
    ("full ext", f2(EX["table2"][0]["ext"]), "from 0.68 over the full panel to 0.81"),
    ("deployed ext", f2(EX["table2"][3]["ext"]), "to 0.81, with a precision of 0.86"),
    ("precision", f2(EX["table2"][3]["precision"]), "precision of 0.86 and a recall of 0.76"),
    ("discarded", f2(EX["discarded_panel_median_ext"]), "discarded drugs sat near 0.63"),
    ("rho>=0.4 recall", f"{EX['table2'][2]['recall']*100:.0f}\\%", "retained 92\\%"),
    ("tertile drops", f"{f1(EX['tertiles']['low']['drop_pts'])} and {f1(EX['tertiles']['mid']['drop_pts'])}", "lost 7.2 and 9.2 AUROC"),
    ("high drop", f1(EX["tertiles"]["high"]["drop_pts"]), "lost 2.3 and reached"),
    ("nested thr", f2(S1["screen_RF_fitted"]["nested_threshold_median"]), "median threshold of 0.48"),
    ("nested acc", f2(S1["screen_RF_fitted"]["nested_heldout_accuracy_median"]), "median accuracy of 0.76"),
    ("internal baseline", f"{f2(EX['baselines']['i']['screen_auc'])}", "ROC-AUC of 0.86"),
    ("internal R2", f"{EX['baselines']['i']['lodo_r2']*100:.0f}\\%", "explained 42\\%"),
    ("gdsc2 dr", f"{f2(EX['baselines']['gdsc2_dr']['screen_auc'])} and {EX['baselines']['gdsc2_dr']['lodo_r2']*100:.0f}\\%", "reached 0.82 and 36\\%"),
    ("ctrp dr", f"{f2(EX['baselines']['ctrp_dr']['screen_auc'])} and {EX['baselines']['ctrp_dr']['lodo_r2']*100:.0f}\\%", "dynamic range 0.67 and 7\\%"),
    ("diff CI", f"{f2(EX['baselines']['conc_minus_gdsc2_dr_screen_auc_ci'][0])} to {f2(EX['baselines']['conc_minus_gdsc2_dr_screen_auc_ci'][1])}", "interval 0.01 to 0.19"),
    ("combo R2", f2(EX["baselines"]["conc_plus_gdsc2_dr_lodo_r2"]), "($R^2$ 0.69)"),
    ("lodo auc", f2(EX["lodo"]["roc_auc"]), "ROC-AUC of 0.88, close"),
    ("lodo r2", f2(EX["lodo"]["r2"]), "$R^2$ of 0.70 (root-mean-square"),
    ("lodo rmse", f1(EX["lodo"]["rmse_pts"]), "error 6.6 AUROC points"),
    ("brier", f"{f2(EX['lodo']['brier'])} against {f3(EX['lodo']['brier_null'])}", "Brier score 0.20 against 0.245"),
    ("loao ctrp", f2(S2["Random Forest"]["CTRP"]["spearman"]), "0.67 with CTRP held out"),
    ("loao gdsc2", f2(S2["Random Forest"]["GDSC2"]["spearman"]), "0.73 with GDSC2 held out"),
    ("loao prism", f2(S2["Random Forest"]["PRISM"]["spearman"]), "0.47 with PRISM held out"),
    ("loao LR", ", ".join(f2(S2["Logistic Regression"][h]["spearman"]) for h in ["CTRP", "GDSC2"]), "0.67, 0.73 and 0.64 for logistic"),
    ("loao pooled", f2(S2["Random Forest"]["pooled"]["spearman"]), "correlation was 0.44 (drug"),
    ("loao pooled CI", f"{f2(S2['Random Forest']['pooled']['drug_cluster_bootstrap_ci'][0])}--{f2(S2['Random Forest']['pooled']['drug_cluster_bootstrap_ci'][1])}", "0.14--0.64"),
    ("pairs pooled", f2(EX["three_assay_pooled"]["spearman"]), "pooled correlation was 0.78 over 198"),
    ("cons LR vs S", f1(lr3["C_minus_S"]["median_pts"]), "by a median of 4.3 AUROC"),
    ("cons LR vs D", f1(lr3["C_minus_D"]["median_pts"]), "by 4.2 points"),
    ("cons LR vs X", f1(lr3["C_minus_X"]["median_pts"]), "extreme-CTRP subset by 2.6"),
    ("cons LR prism", f"{f1(lrp['C_minus_S']['median_pts'])} and {f1(lrp['C_minus_G']['median_pts'])}", "were 2.9 and 2.6 points"),
    ("cons LR cons", f1(S3["Logistic Regression|cons"]["C_minus_S"]["median_pts"]), "training was 3.3 points"),
    ("cons vs avg", f1(lr3["C_minus_A"]["median_pts"]), "difference was 0.8 points"),
    ("RF vs G", f1(rf3["C_minus_G"]["median_pts"]), "GDSC2-only training by 1.9 points (55 of 65"),
    ("RF vs G worse", str(rf3["C_minus_G"]["n_drugs_C_worse"]), "(55 of 65 drugs worse)"),
    ("RF vs A", f1(rf3["C_minus_A"]["median_pts"]), "averaged labels by 1.0 point"),
    ("n cons drugs", str(lr3["C_minus_S"]["n_drugs"]), "Across 65 drugs"),
]
bad = 0
for lab, val, must in checks:
    ok_val = val.lstrip("-") in must
    ok_tex = (must in TEXN) if HAVE_TEX else "n/a"
    status = "OK " if (ok_val and ok_tex) else "FAIL"
    if status == "FAIL":
        bad += 1
    print(f"{status} {lab:18s} computed={val:22s} in_text={ok_tex}")
print(f"\n{len(checks) - bad}/{len(checks)} checks passed"
      + ("" if HAVE_TEX else " (values vs quoted text only; manuscript source not found)"))
