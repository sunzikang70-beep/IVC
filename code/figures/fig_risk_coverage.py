"""Risk-coverage curves (eight unseen T-LESS objects); unified style."""
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
import pandas as pd
OUT=os.environ.get('FIGOUT','/home/claude/newfigs/out')
base=os.environ.get('RCDATA','/mnt/user-data/uploads/codex_paper_workspace/outputs/p0_risk_coverage_20261104/')
c=pd.read_csv(base+"risk_coverage_curves.csv"); fx=pd.read_csv(base+"fixed_candidates_reference.csv")
S={"Geometric cost selector (classical)":("#D55E00","-","Geometric-cost gate"),
   "D joint (ours)":("#56B4E9","--","SARef + residual correction (variant D)"),
   "A ordinary ranking":(C['sar'],"-","SARef")}
fig,axs=plt.subplots(1,2,figsize=(W2,62*MM)); fig.subplots_adjust(left=0.065,right=0.99,top=0.9,bottom=0.3,wspace=0.2)
for key,(col,ls,lab) in S.items():
    d=c[c.ordering==key].sort_values("update_pct")
    for ax,y in zip(axs,["AR_gain_pp","all_target_ADD_regression_pct"]):
        ax.plot(d.update_pct,d[y],color=col,lw=1.3,ls=ls,label=lab,solid_capstyle="round")
        e=d.iloc[-1]; ax.plot(e.update_pct,e[y],"o",ms=4,color=col,mec="white",mew=0.8,zorder=5)
for ax,y in zip(axs,["AR_gain_pp","all_target_ADD_regression_pct"]):
    ax.plot(fx.update_pct,fx[y],"D",ms=3.6,color="#8A9099",mec="white",mew=0.6,zorder=4,label="Fixed refinements (6)",ls="none")
    ax.grid(axis="y",color=C['grid'],lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(0,104); ax.set_xlabel("Conditions updated (%)")
axs[0].set_ylabel("AR gain over input (pp)"); axs[0].set_ylim(0,2.6)
axs[1].set_ylabel("Conditions with higher ADD-S (%)"); axs[1].set_ylim(0,30)
panel_title(axs[0],"(a) AR gain"); panel_title(axs[1],"(b) ADD-S regressions")
dA=c[c.ordering=="A ordinary ranking"].iloc[-1]
axs[0].annotate(f"SARef: +{dA.AR_gain_pp:.2f} pp at {dA.update_pct:.0f}%",(dA.update_pct,dA.AR_gain_pp),xytext=(60,1.55),
                arrowprops=dict(arrowstyle="-",color="#8A9099",lw=0.6))
bf=fx.loc[fx.AR_gain_pp.idxmax()]
axs[0].annotate(f"best fixed: +{bf.AR_gain_pp:.2f} pp",(100,bf.AR_gain_pp),xytext=(95,0.18),ha="right",arrowprops=dict(arrowstyle="-",color="#8A9099",lw=0.6))
axs[1].annotate(f"SARef: {dA.all_target_ADD_regression_pct:.1f}%",(dA.update_pct,dA.all_target_ADD_regression_pct),xytext=(62,12.0),
                arrowprops=dict(arrowstyle="-",color="#8A9099",lw=0.6))
h,l=axs[0].get_legend_handles_labels(); order=[2,1,0,3]
fig.legend([h[i] for i in order],[l[i] for i in order],loc="lower center",ncol=4,bbox_to_anchor=(0.5,0.0),columnspacing=1.4,handlelength=2.0)
save(fig,os.path.join(OUT,'fig_risk_coverage'))
print(round(dA.AR_gain_pp,2),round(dA.update_pct,1),round(dA.all_target_ADD_regression_pct,1),round(bf.AR_gain_pp,2))
