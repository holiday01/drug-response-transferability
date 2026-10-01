#!/usr/bin/env python3
"""Recompute every headline number of the paper from the REV result files and compare
it with the value reported in the paper.
Each check: (label, value recomputed from results/, value reported in the paper)."""
import json
import pandas as pd

from config import RESULTS_DIR
R = str(RESULTS_DIR)
S1 = json.load(open(f"{R}/rev1_independent/summary.json"))
EX = json.load(open(f"{R}/rev1_independent/extra_summary.json"))
S2 = json.load(open(f"{R}/rev2_three_assay/summary.json"))
S3 = json.load(open(f"{R}/rev3_consensus/summary.json"))
RS = json.load(open(f"{R}/rev1b_refsize_single/summary.json"))
_rep = pd.read_csv(f"{R}/rev1b_refsize_single/replicates.csv")
for _k in (20, 50, 100):  # exact values from replicates (summary.json is pre-rounded to 3 dp)
    _x = _rep[_rep.k == _k].spearman
    RS[f"ref{_k}"]["spearman"].update(median=_x.median(), p2_5=_x.quantile(0.025), p97_5=_x.quantile(0.975))
BP = json.load(open(f"{R}/rev2_three_assay/block_permutation.json"))
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
    ("n drugs", str(S1["n_drugs"]), "58"),
    ("RF spearman", f2(rf["spearman"]), "0.85"),
    ("RF CI", f"{f2(rf['ci'][0])}--{f2(rf['ci'][1])}", "0.71--0.92"),
    ("5-model range", f"{f2(min(sp))}--{f2(max(sp))}", "0.85--0.88"),
    ("per-split range", f"{f2(rf['per_split_range'][0])} to {f2(rf['per_split_range'][1])}", "0.74 to 0.86"),
    ("partial RF", f2(rf["partial_internal"][0]), "0.67"),
    ("partial P", f"{rf['partial_internal'][1]:.1e}".replace("e-0", "\\times10^{-").rstrip() + "}", "8.9\\times10^{-9}"),
    ("partial range", f"{f2(min(pa))} to {f2(max(pa))}", "0.67 to 0.71"),
    ("OLS R2 range", f"{min(v['ols_r2'] for k, v in EX['table1'].items() if k != 'Model average')*100:.0f}\\% to {max(v['ols_r2'] for k, v in EX['table1'].items() if k != 'Model average')*100:.0f}\\%", "70\\% to 76\\%"),
    ("leaky", f2(S1["leaky_vs_independent_RF_fitted"]["spearman_leaky"]), "0.85"),
    ("conc diff", f3(S1["leaky_vs_independent_RF_fitted"]["median_abs_diff_concordance"]), "0.006"),
    ("trapz range", f"{f2(min(cv[f'trapz|{m}']['spearman'] for m in M))}--{f2(max(cv[f'trapz|{m}']['spearman'] for m in M))}", "0.83--0.85"),
    ("legacy range", f"{f2(min(cv[f'legacy|{m}']['spearman'] for m in M))}--{f2(max(cv[f'legacy|{m}']['spearman'] for m in M))}", "0.81--0.84"),
    ("summary agreement", f2(S1["gdsc2_summary_agreement"]["median_fitted_vs_legacy"]), "0.90"),
    ("crizotinib", f2(S1["gdsc2_summary_agreement"]["min_fitted_vs_legacy"]), "0.31"),
    ("single ref50", f2(RS["ref50"]["spearman"]["median"]), "0.67"),
    ("single ref50 range", f"{f2(RS['ref50']['spearman']['p2_5'])}--{f2(RS['ref50']['spearman']['p97_5'])}", "0.53--0.77"),
    ("single ref100", f2(RS["ref100"]["spearman"]["median"]), "0.73"),
    ("single ref100 range", f"{f2(RS['ref100']['spearman']['p2_5'])}--{f2(RS['ref100']['spearman']['p97_5'])}", "0.64--0.82"),
    ("single full", f2(float(pd.read_csv(f"{R}/rev1b_refsize_single/full_reference_per_split.csv").spearman.median())), "0.79"),
    ("single ref20", f"{f2(RS['ref20']['spearman']['median'])} ({f2(RS['ref20']['spearman']['p2_5'])}--{f2(RS['ref20']['spearman']['p97_5'])})", "0.53 (0.34--0.68)"),
    ("single screen", f"{f2(RS['ref50']['screen_auc']['median'])} with 50 and {f2(RS['ref100']['screen_auc']['median'])} with 100 reference cell lines, against {f2(RS['full_reference_per_split']['screen_auc']['median'])}", "0.82 with 50 and 0.86 with 100 reference cell lines, against 0.89"),
    ("single agreement", f"{RS['ref50']['call_agreement_with_full']['median']*100:.0f}\\% of drugs with 50 and {RS['ref100']['call_agreement_with_full']['median']*100:.0f}\\%", "81\\% of drugs with 50 and 86\\%"),
    ("block perm RF", f"{BP['Random Forest']['block_permutation_p']:.3f}", "0.006"),
    ("median n ref", str(int(S1["reference_size_sweep_RF_fitted"]["full_reference"]["median_n_ref"])), "333"),
    ("test lines", str(int(S1["test_lines_median"])), "247"),
    ("screen AUC", f2(EX["screen_auc_drug_level"]), "0.91"),
    ("screen per split", f"{f2(S1['screen_RF_fitted']['screen_auc_range'][0])} to {f2(S1['screen_RF_fitted']['screen_auc_range'][1])}", "0.85 to 0.92"),
    ("model avg screen", f2(EX["screen_auc_model_average"]), "0.90"),
    ("n transferable", str(EX["n_transferable"]), "25"),
    ("rho>=0.5 n", str(EX["table2"][3]["n"]), "22"),
    ("full ext", f2(EX["table2"][0]["ext"]), "0.68"),
    ("deployed ext", f2(EX["table2"][3]["ext"]), "0.81"),
    ("precision", f2(EX["table2"][3]["precision"]), "0.86"),
    ("discarded", f2(EX["discarded_panel_median_ext"]), "0.63"),
    ("rho>=0.4 recall", f"{EX['table2'][2]['recall']*100:.0f}\\%", "92\\%"),
    ("tertile drops", f"{f1(EX['tertiles']['low']['drop_pts'])} and {f1(EX['tertiles']['mid']['drop_pts'])}", "7.2 and 9.2"),
    ("high drop", f1(EX["tertiles"]["high"]["drop_pts"]), "2.3"),
    ("nested thr", f2(S1["screen_RF_fitted"]["nested_threshold_median"]), "0.48"),
    ("nested acc", f2(S1["screen_RF_fitted"]["nested_heldout_accuracy_median"]), "0.76"),
    ("internal baseline", f"{f2(EX['baselines']['i']['screen_auc'])}", "0.86"),
    ("internal R2", f"{EX['baselines']['i']['lodo_r2']*100:.0f}\\%", "42\\%"),
    ("gdsc2 dr", f"{f2(EX['baselines']['gdsc2_dr']['screen_auc'])} and {EX['baselines']['gdsc2_dr']['lodo_r2']*100:.0f}\\%", "0.82 and 36\\%"),
    ("ctrp dr", f"{f2(EX['baselines']['ctrp_dr']['screen_auc'])} and {EX['baselines']['ctrp_dr']['lodo_r2']*100:.0f}\\%", "0.67 and 7\\%"),
    ("diff CI", f"{f2(EX['baselines']['conc_minus_gdsc2_dr_screen_auc_ci'][0])} to {f2(EX['baselines']['conc_minus_gdsc2_dr_screen_auc_ci'][1])}", "0.01 to 0.19"),
    ("combo R2", f2(EX["baselines"]["conc_plus_gdsc2_dr_lodo_r2"]), "0.69"),
    ("lodo auc", f2(EX["lodo"]["roc_auc"]), "0.88"),
    ("lodo r2", f2(EX["lodo"]["r2"]), "0.70"),
    ("lodo rmse", f1(EX["lodo"]["rmse_pts"]), "6.6"),
    ("brier", f"{f2(EX['lodo']['brier'])} against {f3(EX['lodo']['brier_null'])}", "0.20 against 0.245"),
    ("loao ctrp", f2(S2["Random Forest"]["CTRP"]["spearman"]), "0.67"),
    ("loao gdsc2", f2(S2["Random Forest"]["GDSC2"]["spearman"]), "0.73"),
    ("loao prism", f2(S2["Random Forest"]["PRISM"]["spearman"]), "0.47"),
    ("loao LR", ", ".join(f2(S2["Logistic Regression"][h]["spearman"]) for h in ["CTRP", "GDSC2"]), "0.67, 0.73"),
    ("loao pooled", f2(S2["Random Forest"]["pooled"]["spearman"]), "0.44"),
    ("loao pooled CI", f"{f2(S2['Random Forest']['pooled']['drug_cluster_bootstrap_ci'][0])}--{f2(S2['Random Forest']['pooled']['drug_cluster_bootstrap_ci'][1])}", "0.14--0.64"),
    ("pairs pooled", f2(EX["three_assay_pooled"]["spearman"]), "0.78"),
    ("cons LR vs S", f1(lr3["C_minus_S"]["median_pts"]), "4.3"),
    ("cons LR vs D", f1(lr3["C_minus_D"]["median_pts"]), "4.2"),
    ("cons LR vs X", f1(lr3["C_minus_X"]["median_pts"]), "2.6"),
    ("cons LR prism", f"{f1(lrp['C_minus_S']['median_pts'])} and {f1(lrp['C_minus_G']['median_pts'])}", "2.9 and 2.6"),
    ("cons LR cons", f1(S3["Logistic Regression|cons"]["C_minus_S"]["median_pts"]), "3.3"),
    ("cons vs avg", f1(lr3["C_minus_A"]["median_pts"]), "0.8"),
    ("RF vs G", f1(rf3["C_minus_G"]["median_pts"]), "-1.9"),
    ("RF vs G worse", str(rf3["C_minus_G"]["n_drugs_C_worse"]), "55"),
    ("RF vs A", f1(rf3["C_minus_A"]["median_pts"]), "-1.0"),
    ("n cons drugs", str(lr3["C_minus_S"]["n_drugs"]), "65"),
]
bad = 0
for lab, val, reported in checks:
    ok = val == reported
    bad += not ok
    print(f"{'OK ' if ok else 'FAIL'} {lab:18s} computed={val:22s} reported={reported}")
print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
