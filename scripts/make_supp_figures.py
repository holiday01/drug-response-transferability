#!/usr/bin/env python3
"""Supplementary figures S1-S9 for the transferability paper. Beautiful, consistent
design; all data from benchmark/results/*.csv (real, seed 42). PDF + 300-dpi PNG.
Authored near text-column width so fonts stay >= 8 pt when placed."""
import numpy as np, pandas as pd, json
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
R=str(RESULTS_DIR)
OUT=str(SUPP_FIG_DIR)
import os; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.size":9.5,"axes.titlesize":10.5,"axes.labelsize":9.5,
    "xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"font.family":"DejaVu Sans"})
# palette
BLUE,RED,GREY,GREEN,ORANGE,PURPLE,TEAL,PINK="#2c6fbb","#c0392b","#8a8a8a","#27893f","#e08a1e","#7b4fa3","#1b9e9e","#c94f9a"
PAIRCOL={"CTRP->GDSC2":BLUE,"CTRP->PRISM":ORANGE,"GDSC2->PRISM":GREEN,
         "GDSC2->CTRP":PURPLE,"PRISM->CTRP":TEAL,"PRISM->GDSC2":PINK}
def save(fig,n): fig.savefig(f"{OUT}/{n}.pdf",bbox_inches="tight"); fig.savefig(f"{OUT}/{n}.png",dpi=300,bbox_inches="tight"); plt.close(fig); print("wrote",n)
def fit(ax,x,y,c):
    sl,ic,rv,_,_=stats.linregress(x,y); xx=np.linspace(np.min(x),np.max(x),40)
    ax.plot(xx,sl*xx+ic,color=RED,lw=1.6,zorder=2)

tp=pd.read_csv(f"{R}/V_threeassay_per_pair.csv")
tp["rho"]=tp["rho"].astype(float)

# ============ S1: three-assay 6-pair grid
fig,ax=plt.subplots(2,3,figsize=(7.4,5.0),constrained_layout=True)
for a,pr in zip(ax.ravel(),["CTRP->GDSC2","GDSC2->CTRP","CTRP->PRISM","PRISM->CTRP","GDSC2->PRISM","PRISM->GDSC2"]):
    s=tp[tp.pair==pr][["rho","ext_Random Forest"]].dropna(); x,y=s["rho"].values,s["ext_Random Forest"].values
    a.scatter(x,y,s=26,color=PAIRCOL[pr],edgecolor="k",linewidth=0.25,alpha=0.85)
    fit(a,x,y,PAIRCOL[pr]); a.axhline(0.5,color=GREY,ls=":",lw=0.8)
    sp=stats.spearmanr(x,y).statistic
    a.set_title(pr.replace("->"," $\\to$ "),fontsize=9.5,fontweight="bold")
    a.text(0.04,0.88,f"$\\rho_s$={sp:.2f}\nn={len(x)}",transform=a.transAxes,fontsize=8.5,
           bbox=dict(boxstyle="round",fc="white",ec=GREY,alpha=.9))
    a.set_xlabel("Concordance"); a.set_ylabel("External AUROC")
fig.suptitle("Concordance predicts transfer across all three assays (CTRP, GDSC2, PRISM)",fontsize=10.5,fontweight="bold")
save(fig,"figS1_threeassay_grid")

# ============ S2: pooled + leave-one-assay-out
loo=pd.read_csv(f"{R}/V_threeassay_loo.csv")
fig,ax=plt.subplots(1,2,figsize=(7.2,3.4),constrained_layout=True)
for pr in PAIRCOL:
    s=tp[tp.pair==pr][["rho","ext_Random Forest"]].dropna()
    ax[0].scatter(s["rho"],s["ext_Random Forest"],s=20,color=PAIRCOL[pr],edgecolor="none",alpha=0.7,label=pr.replace("->","$\\to$"))
s=tp[["rho","ext_Random Forest"]].dropna(); fit(ax[0],s["rho"].values,s["ext_Random Forest"].values,RED)
sp=stats.spearmanr(s["rho"],s["ext_Random Forest"]).statistic
ax[0].axhline(0.5,color=GREY,ls=":",lw=0.8); ax[0].set_xlabel("Cross-assay concordance $\\rho$"); ax[0].set_ylabel("External AUROC")
ax[0].set_title("A",loc="left",fontweight="bold"); ax[0].legend(fontsize=6.6,ncol=2,loc="lower right",framealpha=.9)
ax[0].text(0.04,0.9,f"pooled $\\rho_s$={sp:.2f}\nn={len(s)}",transform=ax[0].transAxes,fontsize=8.5,bbox=dict(boxstyle="round",fc="white",ec=GREY))
x,y=loo["reprod_score"].values,loo["transfer_into"].values
ax[1].scatter(x,y,s=34,color=TEAL,edgecolor="k",linewidth=0.3,alpha=0.85); fit(ax[1],x,y,RED)
ax[1].axhline(0.5,color=GREY,ls=":",lw=0.8)
sp2=stats.spearmanr(x,y)
ax[1].set_xlabel("Reproducibility of the other two assays"); ax[1].set_ylabel("Transfer into held-out assay")
ax[1].set_title("B",loc="left",fontweight="bold")
ax[1].text(0.04,0.9,f"$\\rho_s$={sp2.statistic:.2f}\np={sp2.pvalue:.0e}",transform=ax[1].transAxes,fontsize=8.5,bbox=dict(boxstyle="round",fc="white",ec=GREY))
save(fig,"figS2_pooled_loo")

# ============ S3: permutation null
null=np.load(f"{R}/V_perm_null.npy"); srj=json.load(open(f"{R}/V_stats_rigor_summary.json"))
obs=srj["observed_spearman"]
fig,ax=plt.subplots(figsize=(5.0,3.3),constrained_layout=True)
ax.hist(null,bins=60,color=GREY,alpha=0.7,edgecolor="none")
ax.axvline(obs,color=RED,lw=2.2,label=f"observed = {obs:.2f}")
ax.set_xlabel("Spearman under drug-label permutation"); ax.set_ylabel("Count")
ax.text(0.03,0.9,f"permutation $P$ = {srj['permutation_p']:.0e}\n({srj['n_perm']:,} shuffles)",transform=ax.transAxes,fontsize=9,bbox=dict(boxstyle="round",fc="white",ec=GREY))
ax.legend(loc="upper left",bbox_to_anchor=(0.03,0.75)); ax.set_xlim(-0.5,1.0)
save(fig,"figS3_permutation_null")

# ============ S4: alternative concordance metrics
alt=pd.read_csv(f"{R}/V_alt_concordance_metrics.csv")
fig,ax=plt.subplots(1,3,figsize=(7.4,2.8),constrained_layout=True)
for a,(m,lab,col) in zip(ax,[("spearman","Spearman $\\rho$",BLUE),("pearson_z","Pearson (z)",ORANGE),("kendall","Kendall $\\tau$",GREEN)]):
    s=alt[[m,"ext"]].dropna(); a.scatter(s[m],s["ext"],s=24,color=col,edgecolor="k",linewidth=0.25,alpha=0.85)
    fit(a,s[m].values,s["ext"].values,RED); a.axhline(0.5,color=GREY,ls=":",lw=0.8)
    sp=stats.spearmanr(s[m],s["ext"]).statistic
    a.set_title(lab,fontsize=9.5,fontweight="bold"); a.set_xlabel("Concordance metric"); a.set_ylabel("External AUROC")
    a.text(0.04,0.88,f"$\\rho_s$={sp:.2f}",transform=a.transAxes,fontsize=8.5,bbox=dict(boxstyle="round",fc="white",ec=GREY))
fig.suptitle("Transfer is predicted by every cross-assay concordance metric",fontsize=10,fontweight="bold")
save(fig,"figS4_alt_metrics")

# ============ S5: sensitivity analyses
sn=pd.read_csv(f"{R}/V_sensitivity.csv")
fig,ax=plt.subplots(1,3,figsize=(7.4,3.0),constrained_layout=True)
for a,(cat,title,col) in zip(ax,[("binarization","Label binarization",BLUE),("n_features","Number of features",ORANGE),("selection","Feature selection",GREEN)]):
    g=sn[sn.category==cat]
    a.bar(range(len(g)),g["spearman"],color=col,alpha=0.8,edgecolor="k",linewidth=0.4)
    a.axhline(0.8,color=RED,ls="--",lw=1); a.set_ylim(0,1.0)
    a.set_xticks(range(len(g))); a.set_xticklabels(g["setting"],rotation=25,ha="right",fontsize=8)
    a.set_title(title,fontsize=9.5,fontweight="bold"); a.set_ylabel("Spearman ($\\rho$, transfer)")
    for i,v in enumerate(g["spearman"]): a.text(i,v+0.02,f"{v:.2f}",ha="center",fontsize=7.5)
fig.suptitle("Concordance$\\to$transfer is stable to analysis choices",fontsize=10,fontweight="bold")
save(fig,"figS5_sensitivity")

# ============ S6: cross-assay feature overlap (mechanism)
bio=pd.read_csv(f"{R}/V_biology_per_drug.csv")
fig,ax=plt.subplots(figsize=(5.0,3.6),constrained_layout=True)
s=bio[["rho","feature_jaccard","ext_RF"]].dropna()
sc=ax.scatter(s["rho"],s["feature_jaccard"],c=s["ext_RF"],cmap="viridis",s=42,edgecolor="k",linewidth=0.3,vmin=0.5,vmax=0.9)
fit(ax,s["rho"].values,s["feature_jaccard"].values,RED)
sp=stats.spearmanr(s["rho"],s["feature_jaccard"]).statistic
cb=plt.colorbar(sc,ax=ax,fraction=0.046,pad=0.03); cb.set_label("External AUROC",fontsize=9)
ax.set_xlabel("CTRP–GDSC2 concordance $\\rho$"); ax.set_ylabel("Feature overlap (Jaccard, top-200 genes)")
ax.text(0.04,0.9,f"Spearman={sp:.2f}",transform=ax.transAxes,fontsize=9,bbox=dict(boxstyle="round",fc="white",ec=GREY))
ax.set_title("Concordant drugs are learned from the same genes on both assays",fontsize=9.5,fontweight="bold")
save(fig,"figS6_feature_overlap")

# ============ S7: MOA class
fig,ax=plt.subplots(1,2,figsize=(6.8,3.3),constrained_layout=True)
order=["targeted","cytotoxic","epigenetic","other"]; cols=[BLUE,ORANGE,PURPLE,GREY]
for a,(col,lab) in zip(ax,[("rho","Cross-assay concordance $\\rho$"),("ext_RF","External AUROC")]):
    data=[bio[bio.moa_class==o][col].dropna().values for o in order]
    bp=a.boxplot(data,patch_artist=True,widths=0.6,tick_labels=order)
    for p,c in zip(bp["boxes"],cols): p.set_facecolor(c); p.set_alpha(0.35)
    for med in bp["medians"]: med.set_color("k")
    for i,d in enumerate(data): a.scatter(np.random.default_rng(i).normal(i+1,0.05,len(d)),d,s=12,color=cols[i],alpha=0.6,zorder=3)
    a.set_ylabel(lab); a.tick_params(axis="x",rotation=20)
ax[0].set_title("A",loc="left",fontweight="bold"); ax[1].set_title("B",loc="left",fontweight="bold")
save(fig,"figS7_moa_class")

# ============ S8: biomarker recovery
bm=bio.dropna(subset=["biomarker_rank"]).copy(); bm["biomarker_rank"]=bm["biomarker_rank"].astype(int)
bm=bm.sort_values("biomarker_rank")
fig,ax=plt.subplots(figsize=(5.4,4.6),constrained_layout=True)
yy=np.arange(len(bm)); colr=[GREEN if r<=200 else GREY for r in bm["biomarker_rank"]]
ax.hlines(yy,1,bm["biomarker_rank"],color=colr,lw=1.2,alpha=0.6)
ax.scatter(bm["biomarker_rank"],yy,s=42,color=colr,edgecolor="k",linewidth=0.3,zorder=3)
ax.axvline(200,color=RED,ls="--",lw=1,label="top-200 features")
ax.set_xscale("log"); ax.set_yticks(yy); ax.set_yticklabels([f"{d} ({g.split(';')[0]})" for d,g in zip(bm["drug"],bm["biomarker_genes"])],fontsize=7.5)
ax.set_xlabel("Rank of canonical target gene (point-biserial, log scale)")
ax.legend(loc="lower right"); ax.set_title("Known-biomarker recovery",fontsize=10,fontweight="bold")
save(fig,"figS8_biomarker_recovery")

# ============ S9: per-drug internal vs external, 5 models
ext=pd.read_csv(f"{R}/per_drug_external.csv"); ext=ext[ext.n_ext>0]
MODS=["Logistic Regression","Random Forest","XGBoost","SVM (RBF)","DNN (MLP)"]
fig,ax=plt.subplots(2,3,figsize=(7.4,5.0),constrained_layout=True)
for a,m in zip(ax.ravel(),MODS):
    a.scatter(ext[f"int_{m}"],ext[f"ext_{m}"],c=ext["spearman_ctrp_gdsc2"].astype(float),cmap="viridis",s=26,edgecolor="k",linewidth=0.25,vmin=0,vmax=0.8)
    a.plot([0.5,0.95],[0.5,0.95],"--",color=GREY,lw=0.9); a.axhline(0.5,color=RED,ls=":",lw=0.8)
    a.set_title(m,fontsize=9.5,fontweight="bold"); a.set_xlabel("Internal AUROC"); a.set_ylabel("External AUROC")
    a.set_xlim(0.5,0.95); a.set_ylim(0.4,0.95)
ax.ravel()[5].axis("off")
sm=plt.cm.ScalarMappable(cmap="viridis",norm=plt.Normalize(0,0.8)); sm.set_array([])
cb=fig.colorbar(sm,ax=ax.ravel()[5],fraction=0.5,pad=0.05); cb.set_label("CTRP–GDSC2 concordance $\\rho$",fontsize=9)
fig.suptitle("Per-drug internal vs external AUROC for all five algorithms",fontsize=10,fontweight="bold")
save(fig,"figS9_per_model")
print("all supplementary figures written")
