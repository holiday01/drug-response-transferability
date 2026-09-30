#!/usr/bin/env python3
"""Revision (2026-09-30) supplement builder. Tables S1, S2, S4, S8, S9 and
Figures S1, S2, S9 come from the reference/test design with the fitted GDSC2 AUC
(results/rev*); the remaining items are supporting analyses from the
original single-split design and are labelled as such."""
import json
import pandas as pd, numpy as np
from pathlib import Path

from config import RESULTS_DIR, OUTPUT_DIR
R=str(RESULTS_DIR)
OUT=Path(str(OUTPUT_DIR / "supplementary.tex"))
MODELS=["Logistic Regression","Random Forest","XGBoost","SVM (RBF)","DNN (MLP)"]
SHORT={"Logistic Regression":"LR","Random Forest":"RF","XGBoost":"XGB","SVM (RBF)":"SVM","DNN (MLP)":"MLP"}
def esc(s): return str(s).replace("_","\\_").replace("%","\\%").replace("&","\\&")

def longtable(header,colspec,rows,caption,fontsize="\\small",tabcolsep=None):
    h=" & ".join(header)+" \\\\"; body="\n".join(" & ".join(r)+" \\\\" for r in rows)
    pre=fontsize+(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}" if tabcolsep else "")
    return f"""{{{pre}
\\begin{{longtable}}{{{colspec}}}
\\caption*{{{caption}}}\\\\
\\toprule
{h}
\\midrule
\\endfirsthead
\\multicolumn{{{len(header)}}}{{l}}{{\\emph{{(continued)}}}}\\\\
\\toprule
{h}
\\midrule
\\endhead
\\bottomrule
\\endfoot
{body}
\\end{{longtable}}
}}"""

def smalltable(header,colspec,rows,caption):
    h=" & ".join(header)+" \\\\"; body="\n".join(" & ".join(r)+" \\\\" for r in rows)
    return f"""\\begin{{table}}[h]\\centering
\\caption*{{{caption}}}
\\begin{{tabular}}{{{colspec}}}
\\toprule
{h}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

# ---------- data
ext=pd.read_csv(f"{R}/per_drug_external.csv"); ext=ext[ext["n_ext"]>0].copy()
ext["rho"]=ext["spearman_ctrp_gdsc2"].astype(float); ext=ext.sort_values("rho",ascending=False)
det=pd.read_csv(f"{R}/R5_determinants_per_drug.csv")
cons=pd.read_csv(f"{R}/R3_consensus_per_drug.csv")
bd=pd.read_csv(f"{R}/R6_bidirectional_per_drug.csv"); rg=pd.read_csv(f"{R}/R6_regression_per_drug.csv")
tc=pd.read_csv(f"{R}/R4_tcga_bridge_per_drug.csv")
ta=json.load(open(f"{R}/V_threeassay_summary.json")); sr=json.load(open(f"{R}/V_stats_rigor_summary.json"))
sn=pd.read_csv(f"{R}/V_sensitivity.csv"); bio=pd.read_csv(f"{R}/V_biology_per_drug.csv"); bios=json.load(open(f"{R}/V_biology_summary.json"))
ORIG=" (Supporting analysis, original single-split design; GDSC2 mean-viability summary; concordance on all shared cell lines.)"
r1=pd.read_csv(f"{R}/rev1_independent/per_drug_split.csv")
idm=pd.read_csv(f"{R}/rev1_independent/drug_id_map.csv").set_index("drug")
agr=pd.read_csv(f"{R}/rev1_independent/summary_agreement.csv").set_index("drug")
S1=json.load(open(f"{R}/rev1_independent/summary.json")); EX=json.load(open(f"{R}/rev1_independent/extra_summary.json"))
S2=json.load(open(f"{R}/rev2_three_assay/summary.json"))
r3=pd.read_csv(f"{R}/rev3_consensus/per_drug_split.csv")
agg={"conc":("conc_fitted","mean"),"n_ref":("n_ref_fitted","median"),"n_ext":("n_ext_fitted","median"),"leaky":("conc_fitted_leaky_allshared","mean")}
for m in MODELS: agg[f"int_{m}"]=(f"int_{m}","mean"); agg[f"ext_{m}"]=(f"ext_fitted_{m}","mean")
rv=r1.groupby("drug").agg(**agg).sort_values("conc",ascending=False)

# ---------- Tables S1-S7 (per-drug)
SI=longtable(["Drug","GDSC2 ID","$n$ ref.","$n$ ext.","$\\rho$ ref.","$\\rho$ all","Fitted vs mean"],"lcccccc",
  [[esc(d),str(int(idm.loc[d,"chosen_id"])),f"{r.n_ref:.0f}",f"{r.n_ext:.0f}",f"{r.conc:.3f}",f"{r.leaky:.3f}",f"{agr.loc[d,'fitted_vs_legacy']:.2f}"] for d,r in rv.iterrows()]
  +[["VINCRISTINE","1818","--","--","--","--","no fitted curve"]],
  "\\textbf{Table S1. Drug panel (reference/test design).} GDSC2 compound identifier used (the one with most fitted cell lines when several share a name); median numbers of reference and labelled external test cell lines over ten splits; concordance on reference cell lines (mean over splits) and on all shared cell lines; and the Spearman agreement between the fitted GDSC2 area under the curve and the mean-viability summary used earlier. Vincristine has no fitted curve in GDSC2 release 8.5 and is excluded from the primary analyses.",fontsize="\\small",tabcolsep="4pt")
hdr=["Drug"]+[f"{SHORT[m]} int" for m in MODELS]+[f"{SHORT[m]} ext" for m in MODELS]
SII=longtable(hdr,"l"+"c"*10,
  [[esc(d)]+[f"{r[f'int_{m}']:.3f}" if pd.notna(r[f'int_{m}']) else "--" for m in MODELS]
   +[f"{r[f'ext_{m}']:.3f}" if pd.notna(r[f'ext_{m}']) else "--" for m in MODELS] for d,r in rv.iterrows()],
  "\\textbf{Table S2. Per-drug internal and external AUROC by model} (reference/test design, fitted GDSC2 area under the curve, mean over ten splits). LR, logistic regression; RF, random forest; XGB, XGBoost; SVM, RBF support vector machine; MLP, multilayer perceptron.",
  fontsize="\\scriptsize",tabcolsep="1.8pt")
SIII=longtable(["Drug","$\\rho$","CTRP dyn.\\ range","GDSC2 dyn.\\ range","Lineage entropy"],"lcccc",
  [[esc(r["drug"]),f"{r['rho']:.3f}",f"{r['ctrp_dynrange']:.3f}",
    f"{r['gdsc_dynrange']:.3f}" if pd.notna(r['gdsc_dynrange']) else "--",
    f"{r['lineage_entropy']:.2f}" if pd.notna(r['lineage_entropy']) else "--"] for _,r in det.sort_values("rho",ascending=False).iterrows()],
  "\\textbf{Table S3. Per-drug determinant features:} concordance, dynamic range (s.d.) of the CTRP and GDSC2 sensitivity distributions, and Shannon entropy of the lineages of sensitive cell lines."+ORIG+"")
def g(r,c): return f"{r[c]:.3f}" if pd.notna(r[c]) else "--"
c3=r3.groupby("drug").mean(numeric_only=True).sort_values("n_train_C",ascending=False)
SIV=longtable(["Drug","$n$ cons.","LR S","LR G","LR A","LR C","RF S","RF G","RF A","RF C"],"lccccccccc",
  [[esc(d),f"{r.n_train_C:.0f}"]+[g(r,f"{k}|{m}|gdsc2") for m in ["Logistic Regression","Random Forest"] for k in "SGAC"] for d,r in c3.iterrows()],
  "\\textbf{Table S4. Per-drug label-combination results} on the GDSC2 target of held-out cell lines (mean AUROC over ten splits; thresholds from training cell lines). S, CTRP labels only; G, GDSC2 labels only; A, averaged percentile-rank labels; C, consensus cell lines only; $n$ cons., mean number of consensus training cell lines. Size-matched, extreme-only and PRISM-target results are in the released result files.",fontsize="\\footnotesize",tabcolsep="3pt")
BM=["Logistic Regression","Random Forest","SVM (RBF)"]; BSH={"Logistic Regression":"LR","Random Forest":"RF","SVM (RBF)":"SVM"}
SV=longtable(["Drug","$\\rho$"]+[f"{BSH[m]} {k}" for m in BM for k in ("int","ext")],"lc"+"c"*6,
  [[esc(r["drug"]),f"{float(r['rho']):.3f}" if pd.notna(r['rho']) else "--"]+
   [f"{r[f'{k}_{m}']:.3f}" if pd.notna(r[f'{k}_{m}']) else "--" for m in BM for k in ("int","ext")] for _,r in bd.sort_values("rho",ascending=False).iterrows()],
  "\\textbf{Table S5. Reverse-direction validation} (train GDSC2, test CTRP): per-drug internal (held-out GDSC2) and external (CTRP) AUROC."+ORIG+"",tabcolsep="3pt")
SVI=longtable(["Drug","$\\rho$","$n$ ext","Internal Spearman","External Spearman"],"lcccc",
  [[esc(r["drug"]),f"{float(r['rho']):.3f}" if pd.notna(r['rho']) else "--",str(int(r["n_ext"])),
    f"{r['int_spearman']:.3f}",f"{r['ext_spearman']:.3f}"] for _,r in rg.sort_values("rho",ascending=False).iterrows()],
  "\\textbf{Table S6. Continuous-regression transfer:} per-drug internal and external Spearman between predicted and measured sensitivity."+ORIG+"")
SVII=longtable(["Drug","R/NR","$\\rho$","LR","RF","SVM","RF 95\\% CI"],"lcccccc",
  [[esc(r["drug"]),f"{int(r['n_resp'])}/{int(r['n_nonresp'])}",
    f"{float(r['rho']):.3f}" if pd.notna(r['rho']) else "--",f"{r['patAUROC_Logistic Regression']:.3f}",
    f"{r['patAUROC_Random Forest']:.3f}",f"{r['patAUROC_SVM (RBF)']:.3f}",str(r["patAUROC_RF_ci"])] for _,r in tc.sort_values("patAUROC_mean",ascending=False).iterrows()],
  "\\textbf{Table S7. Patient (TCGA) transfer (exploratory):} per-drug patient-response AUROC by model, with the RF bootstrap CI. R, responder; NR, non-responder.")

# ---------- SVIII: three-assay (REV-2)
pr_rows=[[p.replace("->"," $\\to$ "),str(v["n"]),f"{v['spearman']:+.3f}"] for p,v in EX["three_assay_pairs"].items()]
pr_rows.append(["Pooled (all six pairs)",str(EX["three_assay_pooled"]["n"]),f"{EX['three_assay_pooled']['spearman']:+.3f}"])
for mod in ["Random Forest","Logistic Regression"]:
    for H in ["CTRP","GDSC2","PRISM"]:
        v=S2[mod][H]; pr_rows.append([f"LOAO, {H} held out ({SHORT[mod]})",str(v["n"]),f"{v['spearman']:+.3f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]"])
    v=S2[mod]["pooled"]; pr_rows.append([f"LOAO, pooled ({SHORT[mod]})","99 rows",f"{v['spearman']:+.3f} [{v['drug_cluster_bootstrap_ci'][0]:.2f}, {v['drug_cluster_bootstrap_ci'][1]:.2f}]; perm. $P$={v['within_assay_permutation_p']:.0e}"])
SVIII=smalltable(["Analysis","$n$","Spearman [95\\% CI]"],"lcl",pr_rows,
  "\\textbf{Table S8. Three-assay validation} (33 drugs in CTRP, GDSC2 fitted and PRISM; five splits). Ordered pairs relate concordance on all shared cell lines to external AUROC (random forest). Leave-one-assay-out (LOAO) relates the concordance of two assays to transfer into the held-out third assay; intervals are drug bootstraps, and the pooled interval resamples drugs with their three rows; the permutation shuffles scores across drugs within each held-out assay.")

# ---------- SIX: statistics under the reference/test design
am=sr["alt_metrics"]; cv=S1["concordance_vs_external"]; rf=cv["fitted|Random Forest"]; sw=S1["reference_size_sweep_RF_fitted"]; bl=EX["baselines"]; lo=EX["lodo"]
rng_=lambda k: f"{min(cv[f'{k}|{m}']['spearman'] for m in MODELS):.3f}--{max(cv[f'{k}|{m}']['spearman'] for m in MODELS):.3f}"
rig=[["Spearman, RF (drug bootstrap 95\\% CI)",f"{rf['spearman']:.3f} [{rf['ci'][0]:.3f}, {rf['ci'][1]:.3f}]"],
     ["Spearman, RF, range over ten splits",f"{rf['per_split_range'][0]:.3f}--{rf['per_split_range'][1]:.3f}"],
     ["Spearman, five algorithms, fitted GDSC2 AUC",rng_('fitted')],
     ["Spearman, five algorithms, trapezoid GDSC2 AUC",rng_('trapz')],
     ["Spearman, five algorithms, GDSC2 mean viability",rng_('legacy')],
     ["Partial rank correlation, internal AUROC adjusted (RF)",f"{rf['partial_internal'][0]:.3f} ($P$={rf['partial_internal'][1]:.1e}, df={rf['partial_internal'][2]})"],
     ["Concordance on all shared lines (original design), RF",f"{S1['leaky_vs_independent_RF_fitted']['spearman_leaky']:.3f}"],
     ["Reference set 20 / 50 / 100 lines",f"{sw['ref20']['spearman']:.3f} / {sw['ref50']['spearman']:.3f} / {sw['ref100']['spearman']:.3f}"],
     ["Screen ROC-AUC (in-sample / leave-one-drug-out)",f"{EX['screen_auc_drug_level']:.3f} / {lo['roc_auc']:.3f}"],
     ["Leave-one-drug-out $R^2$, RMSE (pts), Brier (null)",f"{lo['r2']:.3f}, {lo['rmse_pts']:.1f}, {lo['brier']:.3f} ({lo['brier_null']:.3f})"],
     ["Nested threshold, median (held-out accuracy)",f"{S1['screen_RF_fitted']['nested_threshold_median']:.3f} ({S1['screen_RF_fitted']['nested_heldout_accuracy_median']:.3f})"],
     ["Alternative score: internal AUROC (ROC-AUC; LODO $R^2$)",f"{bl['i']['screen_auc']:.3f}; {bl['i']['lodo_r2']:.3f}"],
     ["Alternative score: GDSC2 dynamic range",f"{bl['gdsc2_dr']['screen_auc']:.3f}; {bl['gdsc2_dr']['lodo_r2']:.3f}"],
     ["Alternative score: CTRP dynamic range",f"{bl['ctrp_dr']['screen_auc']:.3f}; {bl['ctrp_dr']['lodo_r2']:.3f}"],
     ["ROC-AUC, concordance minus GDSC2 dynamic range",f"[{bl['conc_minus_gdsc2_dr_screen_auc_ci'][0]:.3f}, {bl['conc_minus_gdsc2_dr_screen_auc_ci'][1]:.3f}]"],
     ["Permutation $P$, original design (20,000 shuffles)",f"{sr['permutation_p']:.1e}"],
     ["Pearson / Kendall concordance metric, original design",f"{am['pearson_z']['spearman_with_ext']:+.3f} / {am['kendall']['spearman_with_ext']:+.3f}"]]
am=sr["alt_metrics"]
SIX=smalltable(["Statistic","Value"],"ll",rig,
  "\\textbf{Table S9. Statistical summary} of the concordance--transfer relationship (CTRP$\\to$GDSC2, 58 drugs, reference/test design unless noted). The partial rank correlation is the Pearson correlation of rank residuals.")

# ---------- SX: sensitivity
sx=[[esc(r["category"]),esc(str(r["setting"])),str(int(r["n"])),f"{r['spearman']:.3f}",f"{r['p']:.1e}",f"{r['median_ext']:.3f}"] for _,r in sn.iterrows()]
SX=smalltable(["Setting type","Value","$n$","Spearman","$P$","median ext."],"llcccc",sx,
  "\\textbf{Table S10. Sensitivity analysis.} Spearman(concordance, external AUROC) under alternative label binarizations, feature counts, and feature-selection methods."+ORIG+"")

# ---------- SXI: biology per-drug
bx=[]
for _,r in bio.sort_values("rho",ascending=False).iterrows():
    bx.append([esc(r["drug"]),f"{r['rho']:.2f}",f"{r['ext_RF']:.3f}",esc(str(r["moa_class"])),
               f"{r['feature_jaccard']:.3f}" if pd.notna(r["feature_jaccard"]) else "--",
               esc(str(r["biomarker_genes"]).split(";")[0]) if pd.notna(r["biomarker_genes"]) else "--",
               str(int(r["biomarker_rank"])) if pd.notna(r["biomarker_rank"]) else "--"])
SXI=longtable(["Drug","$\\rho$","ext RF","MOA class","Feat.\\ Jaccard","Biomarker","Rank"],"lccllcc",bx,
  "\\textbf{Table S11. Biological validation, per drug:} concordance, external AUROC, mechanism-of-action class, cross-assay feature Jaccard (top-200 genes), and the rank of the canonical target gene where curated."+ORIG+"",tabcolsep="4pt")

FIGCAPS=[
 ("figS1_threeassay_grid","\\textbf{Figure S1. Concordance predicts transfer across three assays.} Per-drug external AUROC (random forest, mean over five splits) versus cross-assay concordance (all shared cell lines) for each of the six ordered train$\\to$test pairs among CTRP, GDSC2 (fitted area under the curve) and PRISM; 33 drugs; red line, ordinary least squares; $\\rho_s$, Spearman."),
 ("figS2_pooled_loo","\\textbf{Figure S2. Pooled and leave-one-assay-out.} \\textbf{A} All six pairs pooled ($n$=198), colored by pair. \\textbf{B} Leave-one-assay-out for logistic regression (random forest in main Figure 5A): concordance of the two assays not involving the held-out assay versus transfer into it."),
 ("figS3_permutation_null","\\textbf{Figure S3. Permutation null.} Distribution of Spearman(concordance, external AUROC) under 20,000 random permutations of the drug labels; the observed value lies far in the tail."+ORIG+""),
 ("figS4_alt_metrics","\\textbf{Figure S4. Alternative concordance metrics.} External AUROC versus per-drug concordance computed as Spearman, Pearson (on standardized values), and Kendall $\\tau$."+ORIG+""),
 ("figS5_sensitivity","\\textbf{Figure S5. Sensitivity to analysis choices.} Spearman(concordance, external AUROC) under alternative label binarizations, numbers of selected features, and feature-selection methods; dashed line at 0.80."+ORIG+""),
 ("figS6_feature_overlap","\\textbf{Figure S6. Cross-assay feature overlap.} Per-drug Jaccard overlap of the top-200 genes selected on CTRP versus GDSC2 labels, against concordance, colored by external AUROC."+ORIG+""),
 ("figS7_moa_class","\\textbf{Figure S7. Mechanism-of-action class.} Concordance \\textbf{(A)} and external AUROC \\textbf{(B)} by MOA class; boxes show median and interquartile range, points are drugs."+ORIG+""),
 ("figS8_biomarker_recovery","\\textbf{Figure S8. Known-biomarker recovery.} Rank of each drug's canonical target gene among point-biserial features (log scale); green marks genes within the top-200."+ORIG+""),
 ("figS9_per_model","\\textbf{Figure S9. Per-drug internal versus external AUROC} for all five algorithms (reference/test design, mean over ten splits), colored by reference concordance."),
]
figs="\n".join(f"""\\begin{{figure}}[h]\\centering
\\includegraphics[width={0.62 if f.startswith('figS3') or f.startswith('figS6') or f.startswith('figS8') else 0.98}\\linewidth]{{supp_figures/{f}.pdf}}
\\caption*{{{c}}}
\\end{{figure}}
\\clearpage""" for f,c in FIGCAPS)

tables="\n\\clearpage\n".join([SI,SII,SIII,SIV,SV,SVI,SVII,SVIII,SIX,SX,SXI])

doc=f"""\\documentclass[11pt]{{article}}
\\usepackage[T1]{{fontenc}}
\\usepackage[a4paper,margin=0.9in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
\\usepackage{{caption}}
\\captionsetup{{labelformat=empty,justification=raggedright,singlelinecheck=false,font=small}}
\\renewcommand{{\\arraystretch}}{{1.05}}
\\setlength{{\\LTcapwidth}}{{\\textwidth}}
\\begin{{document}}
\\begin{{center}}
{{\\large\\textbf{{Supplementary Information}}}}\\\\[4pt]
Cross-assay concordance predicts the cross-dataset transferability of
cancer drug-response models\\\\[2pt]
Yen-Jung Chiu
\\end{{center}}
\\vspace{{0.5em}}
\\noindent This document contains Supplementary Figures S1--S9 and Supplementary
Tables S1--S11.
\\clearpage

\\section*{{Supplementary Figures}}
{figs}

\\section*{{Supplementary Tables}}
{tables}

\\end{{document}}
"""
OUT.write_text(doc)
print("wrote supplementary.tex")
print(f"figures: {len(FIGCAPS)} ; tables: SI-SXI (11)")
