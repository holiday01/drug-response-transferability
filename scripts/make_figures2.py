#!/usr/bin/env python3
"""New figures for the expanded paper: Fig5 robustness+prospective, Fig6 TCGA."""
import numpy as np, pandas as pd, json
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import (CCLE_DIR, PRISM_AUC, DRUG_CATALOG, TCGA_EXPR, TCGA_DRUG,
                    RESULTS_DIR, OUTPUT_DIR, FIG_DIR, SUPP_FIG_DIR)
R=str(RESULTS_DIR)
OUT=str(FIG_DIR)
plt.rcParams.update({"font.size":9.5,"axes.titlesize":10,"axes.labelsize":9.5,
    "xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"font.family":"DejaVu Sans"})
BLUE,RED,GREY,GREEN,ORANGE,PURPLE="#2c6fbb","#c0392b","#8a8a8a","#27893f","#e08a1e","#7b4fa3"
def save(fig,n): fig.savefig(f"{OUT}/{n}.pdf"); fig.savefig(f"{OUT}/{n}.png",dpi=300); plt.close(fig); print("wrote",n)

# ===================== FIG 5 : robustness + prospective screen
bd=pd.read_csv(f"{R}/R6_bidirectional_per_drug.csv"); bd["rho"]=bd["rho"].astype(float)
rg=pd.read_csv(f"{R}/R6_regression_per_drug.csv"); rg["rho"]=rg["rho"].astype(float)
cal=pd.read_csv(f"{R}/R2b_screen_calibration.csv")
lodo=json.load(open(f"{R}/R2b_screen_validation.json"))
fig,ax=plt.subplots(1,3,figsize=(6.5,2.7),constrained_layout=True)
# A bidirectional
s=bd[["rho","ext_Random Forest"]].dropna(); x,y=s["rho"].values,s["ext_Random Forest"].values
sl,ic,rv,_,_=stats.linregress(x,y); sp=stats.spearmanr(x,y).statistic
ax[0].scatter(x,y,s=40,color=PURPLE,edgecolor="k",linewidth=0.3,alpha=0.85)
ax[0].plot(np.linspace(x.min(),x.max(),40),sl*np.linspace(x.min(),x.max(),40)+ic,color=RED,lw=1.8)
ax[0].axhline(0.5,color=GREY,ls=":",lw=0.9)
ax[0].set_xlabel("Concordance $\\rho$"); ax[0].set_ylabel("External AUROC (CTRP)")
ax[0].set_title("A",loc="left",fontweight="bold",fontsize=9)
ax[0].text(0.03,0.90,f"Spearman = {sp:.2f}",transform=ax[0].transAxes,fontsize=9,
           bbox=dict(boxstyle="round",fc="white",ec=GREY))
# B regression
s=rg[["rho","ext_spearman"]].dropna(); x,y=s["rho"].values,s["ext_spearman"].values
sl,ic,rv,_,_=stats.linregress(x,y); sp=stats.spearmanr(x,y).statistic
ax[1].scatter(x,y,s=40,color=ORANGE,edgecolor="k",linewidth=0.3,alpha=0.85)
ax[1].plot(np.linspace(x.min(),x.max(),40),sl*np.linspace(x.min(),x.max(),40)+ic,color=RED,lw=1.8)
ax[1].axhline(0,color=GREY,ls=":",lw=0.9)
ax[1].set_xlabel("Concordance $\\rho$"); ax[1].set_ylabel("Transfer (Spearman)")
ax[1].set_title("B",loc="left",fontweight="bold",fontsize=9)
ax[1].text(0.03,0.90,f"Spearman = {sp:.2f}",transform=ax[1].transAxes,fontsize=9,
           bbox=dict(boxstyle="round",fc="white",ec=GREY))
# C calibration (prospective)
xb=np.arange(len(cal))
ax[2].bar(xb,cal["transfer_rate"],color=GREEN,alpha=0.8,edgecolor="k",linewidth=0.4)
for i,r in cal.iterrows(): ax[2].text(i,r.transfer_rate+0.03,f"{r.transfer_rate:.0%}",ha="center",fontsize=8.5)
ax[2].set_xticks(xb); ax[2].set_xticklabels(cal["bin"],rotation=20,fontsize=9)
ax[2].set_ylim(0,1.12); ax[2].set_ylabel("Observed transfer rate")
ax[2].set_xlabel("Concordance $\\rho$ bin")
ax[2].set_title("C",loc="left",fontweight="bold",fontsize=9)
ax[2].text(0.03,0.90,f"LODO ROC-AUC = {lodo['lodo_roc_auc']:.2f}",transform=ax[2].transAxes,fontsize=9,
           bbox=dict(boxstyle="round",fc="white",ec=GREY))
save(fig,"fig5_robustness")

# ===================== FIG 6 : TCGA patient bridge
t=pd.read_csv(f"{R}/R4_tcga_bridge_per_drug.csv"); t["rho"]=t["rho"].astype(float)
t=t.sort_values("patAUROC_mean")
fig,ax=plt.subplots(1,2,figsize=(6.5,3.1),constrained_layout=True)
# A forest plot: RF AUROC with CI
lo=np.array([float(c.split("-")[0]) for c in t["patAUROC_RF_ci"]])
hi=np.array([float(c.split("-")[1]) for c in t["patAUROC_RF_ci"]])
yv=np.arange(len(t)); rf=t["patAUROC_Random Forest"].values
ax[0].errorbar(rf,yv,xerr=[rf-lo,hi-rf],fmt="o",color=BLUE,ecolor=GREY,capsize=3,ms=6)
ax[0].axvline(0.5,color=RED,ls="--",lw=1)
ax[0].set_yticks(yv); ax[0].set_yticklabels([f"{d.title()} (n={n})" for d,n in zip(t["drug"],t["n_pat"])],fontsize=9)
ax[0].set_xlabel("Patient response AUROC")
ax[0].set_title("A",loc="left",fontweight="bold",fontsize=9)
ax[0].set_xlim(0.2,0.85)
# B patient AUROC vs concordance
s=t[["rho","patAUROC_mean","drug"]].dropna()
ax[1].scatter(s["rho"],s["patAUROC_mean"],s=70,color=GREEN,edgecolor="k",linewidth=0.4,zorder=3)
for _,r in s.iterrows(): ax[1].annotate(r["drug"].title(),(r["rho"],r["patAUROC_mean"]),
    fontsize=9,xytext=(3,3),textcoords="offset points")
ax[1].axhline(0.5,color=RED,ls="--",lw=1); ax[1].set_xlim(-0.20,0.90); ax[1].margins(y=0.14)
sp=stats.spearmanr(s["rho"],s["patAUROC_mean"])
ax[1].set_xlabel("Concordance $\\rho$"); ax[1].set_ylabel("Patient AUROC (mean)")
ax[1].set_title("B",loc="left",fontweight="bold",fontsize=9)
ax[1].text(0.97,0.05,f"Spearman = {sp.statistic:+.2f} (n.s.)",va="bottom",ha="right",transform=ax[1].transAxes,fontsize=9,
           bbox=dict(boxstyle="round",fc="white",ec=GREY))
save(fig,"fig6_tcga_bridge")
print("done")
