"""Object-level AR change over the input: fixed refinement (filtered P2P, cap 1) vs SARef."""
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from figstyle import *; from common_data import *
import numpy as np, json
OUT=os.environ.get('FIGOUT','/home/claude/newfigs/out')
fig=plt.figure(figsize=(W2,52*MM))
gs=fig.add_gridspec(1,2,width_ratios=[8+2.5,21+2.5],wspace=0.08,left=0.06,right=0.995,top=0.88,bottom=0.33)
vals={}
for k,(name,title) in enumerate([('tless','(a) T-LESS, 8 unseen objects'),('ycbv','(b) YCB-V, 21 objects')]):
    d=load(name); g=d.groupby('object_id')[['d_'+FIXED,'d_'+OURS]].mean()
    ax=fig.add_subplot(gs[0,k]); x=np.arange(len(g))
    for i,(o,r) in enumerate(g.iterrows()):
        ax.plot([i,i],[r['d_'+FIXED],r['d_'+OURS]],color='#B8BEC6',lw=1.0,zorder=1)
    ax.scatter(x,g['d_'+FIXED],s=13,color=C['fix'],marker='s',zorder=3,label='Fixed refinement (filtered P2P, cap 1)',edgecolors='white',linewidths=0.4)
    ax.scatter(x,g['d_'+OURS],s=15,color=C['sar'],zorder=4,label='SARef',edgecolors='white',linewidths=0.4)
    ax.axhline(0,color=C['ink2'],lw=0.6,zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([str(int(o)) for o in g.index]); ax.set_xlim(-0.6,len(g)-0.4)
    ax.set_xlabel('Object ID'); ax.grid(axis='y',color=C['grid'],lw=0.6); ax.set_axisbelow(True)
    panel_title(ax,title)
    if k==0: ax.set_ylabel('AR change over input (pp)')
    vals[name]={int(o):[round(float(r['d_'+FIXED]),3),round(float(r['d_'+OURS]),3)] for o,r in g.iterrows()}
h,l=ax.get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=2,bbox_to_anchor=(0.5,0.0),columnspacing=1.6,handletextpad=0.3)
save(fig,os.path.join(OUT,'fig_per_object'))
json.dump(vals,open(os.path.join(OUT,'fig_per_object_values.json'),'w'),indent=0)
print(vals)
