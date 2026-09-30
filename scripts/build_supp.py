#!/usr/bin/env python3
"""Build a large supplementary document: 9 supplementary figures + Tables S1-S11,
all from the real result files."""
import json
import pandas as pd, numpy as np
from pathlib import Path

from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
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

# ---------- Tables S1-S7 (per-drug)
SI=longtable(["Drug","$n$ CTRP","$n$ GDSC2 ext.","Concordance $\\rho$"],"lccc",
  [[esc(r["drug"]),str(int(r["n_ctrp"])),str(int(r["n_ext"])),f"{r['rho']:.3f}"] for _,r in ext.iterrows()],
  "\\textbf{Table S1. Drug panel.} Per-drug CTRP-labelled cell lines, independent GDSC2 external cell lines, and CTRP--GDSC2 concordance, sorted by concordance.")
hdr=["Drug"]+[f"{SHORT[m]} int" for m in MODELS]+[f"{SHORT[m]} ext" for m in MODELS]
SII=longtable(hdr,"l"+"c"*10,
  [[esc(r["drug"])]+[f"{r[f'int_{m}']:.3f}" if pd.notna(r[f'int_{m}']) else "--" for m in MODELS]
   +[f"{r[f'ext_{m}']:.3f}" if pd.notna(r[f'ext_{m}']) else "--" for m in MODELS] for _,r in ext.iterrows()],
  "\\textbf{Table S2. Per-drug internal and external AUROC by model.} LR, logistic regression; RF, random forest; XGB, XGBoost; SVM, RBF support vector machine; MLP, multilayer perceptron.",
  fontsize="\\scriptsize",tabcolsep="1.8pt")
SIII=longtable(["Drug","$\\rho$","CTRP dyn.\\ range","GDSC2 dyn.\\ range","Lineage entropy"],"lcccc",
  [[esc(r["drug"]),f"{r['rho']:.3f}",f"{r['ctrp_dynrange']:.3f}",
    f"{r['gdsc_dynrange']:.3f}" if pd.notna(r['gdsc_dynrange']) else "--",
    f"{r['lineage_entropy']:.2f}" if pd.notna(r['lineage_entropy']) else "--"] for _,r in det.sort_values("rho",ascending=False).iterrows()],
  "\\textbf{Table S3. Per-drug determinant features:} concordance, dynamic range (s.d.) of the CTRP and GDSC2 sensitivity distributions, and Shannon entropy of the lineages of sensitive cell lines.")
def g(r,c): return f"{r[c]:.3f}" if pd.notna(r[c]) else "--"
SIV=longtable(["Drug","$n$ cons.","$\\rho$","LR single","LR cons.","RF single","RF cons."],"lcccccc",
  [[esc(r["drug"]),str(int(r["n_train_cons"])),f"{r['rho']:.2f}",g(r,"S_Logistic Regression|cons"),
    g(r,"C_Logistic Regression|cons"),g(r,"S_Random Forest|cons"),g(r,"C_Random Forest|cons")] for _,r in cons.sort_values("rho",ascending=False).iterrows()],
  "\\textbf{Table S4. Per-drug consensus results} on the reproducible target: number of concordant training cell lines, concordance, and AUROC under single-assay and consensus training for LR and RF.")
BM=["Logistic Regression","Random Forest","SVM (RBF)"]; BSH={"Logistic Regression":"LR","Random Forest":"RF","SVM (RBF)":"SVM"}
SV=longtable(["Drug","$\\rho$"]+[f"{BSH[m]} {k}" for m in BM for k in ("int","ext")],"lc"+"c"*6,
  [[esc(r["drug"]),f"{float(r['rho']):.3f}" if pd.notna(r['rho']) else "--"]+
   [f"{r[f'{k}_{m}']:.3f}" if pd.notna(r[f'{k}_{m}']) else "--" for m in BM for k in ("int","ext")] for _,r in bd.sort_values("rho",ascending=False).iterrows()],
  "\\textbf{Table S5. Reverse-direction validation} (train GDSC2, test CTRP): per-drug internal (held-out GDSC2) and external (CTRP) AUROC.",tabcolsep="3pt")
SVI=longtable(["Drug","$\\rho$","$n$ ext","Internal Spearman","External Spearman"],"lcccc",
  [[esc(r["drug"]),f"{float(r['rho']):.3f}" if pd.notna(r['rho']) else "--",str(int(r["n_ext"])),
    f"{r['int_spearman']:.3f}",f"{r['ext_spearman']:.3f}"] for _,r in rg.sort_values("rho",ascending=False).iterrows()],
  "\\textbf{Table S6. Continuous-regression transfer:} per-drug internal and external Spearman between predicted and measured sensitivity.")
SVII=longtable(["Drug","R/NR","$\\rho$","LR","RF","SVM","RF 95\\% CI"],"lcccccc",
  [[esc(r["drug"]),f"{int(r['n_resp'])}/{int(r['n_nonresp'])}",
    f"{float(r['rho']):.3f}" if pd.notna(r['rho']) else "--",f"{r['patAUROC_Logistic Regression']:.3f}",
    f"{r['patAUROC_Random Forest']:.3f}",f"{r['patAUROC_SVM (RBF)']:.3f}",str(r["patAUROC_RF_ci"])] for _,r in tc.sort_values("patAUROC_mean",ascending=False).iterrows()],
  "\\textbf{Table S7. Patient (TCGA) transfer:} per-drug patient-response AUROC by model, with the RF bootstrap CI. R, responder; NR, non-responder.")

# ---------- SVIII: three-assay per-pair
pr_rows=[]
for pr,d in ta["pairs"].items():
    pr_rows.append([pr.replace("->"," $\\to$ "),str(d["n"]),f"{d['median_rho']:.2f}",f"{d['median_ext']:.3f}",f"{d['spearman']:+.3f}"])
pr_rows.append(["\\textbf{Pooled (all 6)}",str(ta["pooled_n"]),"--","--",f"\\textbf{{{ta['pooled_spearman']:+.3f}}}"])
pr_rows.append(["Leave-one-assay-out",str(ta["loo_n"]),"--","--",f"{ta['loo_spearman']:+.3f}"])
SVIII=smalltable(["Train $\\to$ test","$n$ drugs","median $\\rho$","median ext.\\ AUROC","Spearman($\\rho$,ext)"],"lcccc",pr_rows,
  "\\textbf{Table S8. Three-assay validation.} Concordance predicts external AUROC (random forest) for every ordered pair of CTRP, GDSC2 and PRISM, pooled across pairs, and for a drug's mean reproducibility of the other two assays predicting transfer into a held-out third assay (leave-one-assay-out).")

# ---------- SIX: statistical rigor
am=sr["alt_metrics"]
rig=[["Observed Spearman (RF, CTRP$\\to$GDSC2)",f"{sr['observed_spearman']:.3f}"],
     [f"Permutation $P$ ({sr['n_perm']:,} shuffles)",f"{sr['permutation_p']:.1e}"],
     ["Bootstrap 95\\% CI",f"[{sr['boot_ci'][0]:.3f}, {sr['boot_ci'][1]:.3f}]"],
     ["Spearman metric $\\to$ transfer",f"{am['spearman']['spearman_with_ext']:+.3f}"],
     ["Pearson (z) metric $\\to$ transfer",f"{am['pearson_z']['spearman_with_ext']:+.3f}"],
     ["Kendall $\\tau$ metric $\\to$ transfer",f"{am['kendall']['spearman_with_ext']:+.3f}"],
     ["Partial (control $n$, lineage, difficulty)",f"{sr['partial_spearman_full']:+.3f} ($P$={sr['partial_p']:.1e})"]]
SIX=smalltable(["Statistic","Value"],"ll",rig,
  "\\textbf{Table S9. Statistical robustness} of the concordance--transfer relationship (CTRP$\\to$GDSC2, $n$=59 drugs).")

# ---------- SX: sensitivity
sx=[[esc(r["category"]),esc(str(r["setting"])),str(int(r["n"])),f"{r['spearman']:.3f}",f"{r['p']:.1e}",f"{r['median_ext']:.3f}"] for _,r in sn.iterrows()]
SX=smalltable(["Setting type","Value","$n$","Spearman","$P$","median ext."],"llcccc",sx,
  "\\textbf{Table S10. Sensitivity analysis.} Spearman(concordance, external AUROC) under alternative label binarizations, feature counts, and feature-selection methods.")

# ---------- SXI: biology per-drug
bx=[]
for _,r in bio.sort_values("rho",ascending=False).iterrows():
    bx.append([esc(r["drug"]),f"{r['rho']:.2f}",f"{r['ext_RF']:.3f}",esc(str(r["moa_class"])),
               f"{r['feature_jaccard']:.3f}" if pd.notna(r["feature_jaccard"]) else "--",
               esc(str(r["biomarker_genes"]).split(";")[0]) if pd.notna(r["biomarker_genes"]) else "--",
               str(int(r["biomarker_rank"])) if pd.notna(r["biomarker_rank"]) else "--"])
SXI=longtable(["Drug","$\\rho$","ext RF","MOA class","Feat.\\ Jaccard","Biomarker","Rank"],"lccllcc",bx,
  "\\textbf{Table S11. Biological validation, per drug:} concordance, external AUROC, mechanism-of-action class, cross-assay feature Jaccard (top-200 genes), and the rank of the canonical target gene where curated.",tabcolsep="4pt")

FIGCAPS=[
 ("figS1_threeassay_grid","\\textbf{Figure S1. Concordance predicts transfer across three assays.} Per-drug external AUROC (random forest) versus cross-assay concordance for each of the six ordered train$\\to$test pairs among CTRP, GDSC2 and PRISM; red line, ordinary least squares; $\\rho_s$, Spearman."),
 ("figS2_pooled_loo","\\textbf{Figure S2. Pooled and leave-one-assay-out.} \\textbf{A} All six pairs pooled ($n$=439), colored by pair. \\textbf{B} A drug's mean reproducibility across the two assays not being transferred into, versus its transfer into the held-out third assay."),
 ("figS3_permutation_null","\\textbf{Figure S3. Permutation null.} Distribution of Spearman(concordance, external AUROC) under 20,000 random permutations of the drug labels; the observed value lies far in the tail."),
 ("figS4_alt_metrics","\\textbf{Figure S4. Alternative concordance metrics.} External AUROC versus per-drug concordance computed as Spearman, Pearson (on standardized values), and Kendall $\\tau$."),
 ("figS5_sensitivity","\\textbf{Figure S5. Sensitivity to analysis choices.} Spearman(concordance, external AUROC) under alternative label binarizations, numbers of selected features, and feature-selection methods; dashed line at 0.80."),
 ("figS6_feature_overlap","\\textbf{Figure S6. Cross-assay feature overlap.} Per-drug Jaccard overlap of the top-200 genes selected on CTRP versus GDSC2 labels, against concordance, colored by external AUROC."),
 ("figS7_moa_class","\\textbf{Figure S7. Mechanism-of-action class.} Concordance \\textbf{(A)} and external AUROC \\textbf{(B)} by MOA class; boxes show median and interquartile range, points are drugs."),
 ("figS8_biomarker_recovery","\\textbf{Figure S8. Known-biomarker recovery.} Rank of each drug's canonical target gene among point-biserial features (log scale); green marks genes within the top-200."),
 ("figS9_per_model","\\textbf{Figure S9. Per-drug internal versus external AUROC} for all five algorithms, colored by CTRP--GDSC2 concordance."),
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
