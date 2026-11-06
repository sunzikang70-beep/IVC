import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
base="/mnt/user-data/uploads/codex_paper_workspace/outputs/p0_risk_coverage_20261104/"
c=pd.read_csv(base+"risk_coverage_curves.csv"); fx=pd.read_csv(base+"fixed_candidates_reference.csv")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.edgecolor":"#8a8985","axes.labelcolor":"#0b0b0b","xtick.color":"#52514e","ytick.color":"#52514e"})
S={"Geometric cost selector (classical)":("#1baf7a","Classical geometric-cost gate"),
   "A ordinary ranking":("#eb6834","Ordinary predicted-gain ranking"),
   "D joint (ours)":("#2a78d6","State-conditioned residual selection (ours)")}
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),sharex=True,constrained_layout=True)
for key,(col,lab) in S.items():
    d=c[c.ordering==key].sort_values("update_pct")
    for ax,y in zip(axs,["AR_gain_pp","all_target_ADD_regression_pct"]):
        ax.plot(d.update_pct,d[y],color=col,lw=2,label=lab,solid_capstyle="round")
        e=d.iloc[-1]; ax.plot(e.update_pct,e[y],"o",ms=6,color=col,mec="white",mew=1.5,zorder=5)
for ax,y in zip(axs,["AR_gain_pp","all_target_ADD_regression_pct"]):
    ax.plot(fx.update_pct,fx[y],"D",ms=5,color="#8a8985",mec="white",mew=1,zorder=4,label="Fixed P2P / filtered P2P (6 steps)")
    ax.grid(axis="y",color="#e7e6e2",lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(0,104); ax.set_xlabel("Targets updated (%)")
axs[0].set_ylabel("AR gain over initializer (pp)"); axs[0].set_ylim(0,2.6)
axs[1].set_ylabel("Targets with worse ADD-S (%)"); axs[1].set_ylim(0,30)
axs[0].set_title("a  Accuracy",loc="left",fontsize=9.5,fontweight="bold",color="#0b0b0b")
axs[1].set_title("b  Regression risk",loc="left",fontsize=9.5,fontweight="bold",color="#0b0b0b")
# selective direct labels
dD=c[c.ordering=="D joint (ours)"].iloc[-1]
axs[0].annotate(f"ours: +{dD.AR_gain_pp:.2f} pp at {dD.update_pct:.0f}%",(dD.update_pct,dD.AR_gain_pp),xytext=(52,1.55),
                fontsize=8,color="#0b0b0b",arrowprops=dict(arrowstyle="-",color="#8a8985",lw=0.8))
bf=fx.loc[fx.AR_gain_pp.idxmax()]
axs[0].annotate(f"best fixed: +{bf.AR_gain_pp:.2f} pp",(100,bf.AR_gain_pp),xytext=(95,0.18),ha="right",fontsize=8,color="#0b0b0b",
                arrowprops=dict(arrowstyle="-",color="#8a8985",lw=0.8))
axs[1].annotate(f"ours: {dD.all_target_ADD_regression_pct:.1f}%",(dD.update_pct,dD.all_target_ADD_regression_pct),xytext=(50,1.0),
                fontsize=8,color="#0b0b0b",arrowprops=dict(arrowstyle="-",color="#8a8985",lw=0.8))
dA=c[c.ordering=="A ordinary ranking"].iloc[-1]
axs[1].annotate(f"ranking: {dA.all_target_ADD_regression_pct:.1f}%",(dA.update_pct,dA.all_target_ADD_regression_pct),xytext=(60,10.5),
                fontsize=8,color="#0b0b0b",arrowprops=dict(arrowstyle="-",color="#8a8985",lw=0.8))
h,l=axs[0].get_legend_handles_labels()
order=[2,1,0,3]; h=[h[i] for i in order]; l=[l[i] for i in order]
fig.legend(h,l,loc="outside lower center",ncol=2,frameon=False,fontsize=8)
for ext in ["png","pdf"]:
    fig.savefig(f"/tmp/p0fig/fig_risk_coverage_unseen8.{ext}",dpi=300)
