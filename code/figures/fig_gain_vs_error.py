"""AR change vs initial ADD-S error (in diameters), pooled per bin; 95% object-cluster bootstrap bands."""
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from figstyle import *; from common_data import *
import numpy as np
OUT=os.environ.get('FIGOUT','/home/claude/newfigs/out')
labs=['<1.5','1.5–2.5','2.5–4','4–6','6–10','10–20','20–100','≥100']
fig=plt.figure(figsize=(W2,66*MM))
gs=fig.add_gridspec(2,2,height_ratios=[3.2,1],hspace=0.08,wspace=0.16,left=0.06,right=0.995,top=0.93,bottom=0.27)
res={}
for k,(name,title) in enumerate([('tless','(a) T-LESS, 8 unseen objects'),('ycbv','(b) YCB-V, 21 objects')]):
    d=load(name); ax=fig.add_subplot(gs[0,k]); axc=fig.add_subplot(gs[1,k],sharex=ax)
    x=np.arange(len(labs))
    for col,c,lab,ls,mk in [('d_'+FIXED,C['fix'],'Fixed refinement (filtered P2P, cap 1)','-','s'),('d_'+OURS,C['sar'],'SARef','-','o'),('d_veto',C['veto'],'Oracle veto','--',None)]:
        m,lo,hi,cnt=binstats(d,col); res[(name,col)]=(m,lo,hi,cnt)
        if mk: ax.fill_between(x,lo,hi,color=c,alpha=0.14,lw=0)
        ax.plot(x,m,ls=ls,color=c,marker=mk,ms=3.2,mec='white',mew=0.5,label=lab,zorder=3)
    ax.axhline(0,color=C['ink2'],lw=0.6,zorder=1)
    ax.grid(axis='y',color=C['grid'],lw=0.6); ax.set_axisbelow(True)
    panel_title(ax,title); ax.tick_params(labelbottom=False)
    if k==0: ax.set_ylabel('AR change over input (pp)')
    axc.bar(x,cnt,width=0.62,color='#C9CED6',lw=0)
    axc.set_xticks(x); axc.set_xticklabels(labs)
    axc.set_xlabel('Initial ADD-S error (% of object diameter)')
    if k==0: axc.set_ylabel('Conditions',labelpad=7)
    step=1000 if cnt.max()>2000 else 100; axc.set_yticks([0,int(cnt.max()//step*step)])
    axc.spines['left'].set_bounds(0,axc.get_yticks()[-1])
h,l=ax.get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=3,bbox_to_anchor=(0.5,0.0),handlelength=2.2,columnspacing=1.6)
save(fig,os.path.join(OUT,'fig_gain_vs_error'))
import json
json.dump({f'{n}|{c}':[list(map(lambda v: None if v!=v else round(float(v),3),a)) for a in r] for (n,c),r in res.items()},open(os.path.join(OUT,'fig_gain_vs_error_values.json'),'w'),indent=0)
print('ok')
