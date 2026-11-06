"""Which pose SARef returns, by initialization regime."""
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from figstyle import *; from common_data import *
import numpy as np, json
from matplotlib.patches import Patch
OUT=os.environ.get('FIGOUT','/home/claude/newfigs/out')
acts=['protocolB_e2e_initial_pose','protocolF_se3_trust_p2p_s025','protocolF_se3_trust_p2p_s050','protocolF_se3_trust_p2p_s100',
      'protocolF_se3_trust_filtered_s025','protocolF_se3_trust_filtered_s050','protocolF_se3_trust_filtered_s100']
labs=['keep $T_0$','P2P, cap ¼','P2P, cap ½','P2P, cap 1','filtered, cap ¼','filtered, cap ½','filtered, cap 1']
cols=[C['keep']]+C['cand']
fig,ax=plt.subplots(figsize=(W1,62*MM)); fig.subplots_adjust(left=0.2,right=0.98,top=0.97,bottom=0.38)
rows=[]; ylab=[]; y=0; ys=[]; shares={}
for name,dn in [('tless','T-LESS'),('ycbv','YCB-V')]:
    d=load(name)
    for r in ['native','mild','hard']:
        s=d.loc[d.init_regime.eq(r),'A_selected_action'].value_counts(normalize=True).reindex(acts,fill_value=0)*100
        shares[f'{dn}|{r}']=[round(float(v),1) for v in s]
        left=0
        for v,c in zip(s,cols):
            ax.barh(y,v,left=left,color=c,height=0.72,lw=0)
            if v>=9: ax.text(left+v/2,y,f'{v:.0f}',ha='center',va='center',color='white' if c in (C['cand'][1],C['cand'][2],C['cand'][5]) else C['ink'],fontsize=7)
            left+=v
        ys.append(y); ylab.append(f'{dn}, {r}'); y+=1
    y+=0.5
ax.set_yticks(ys); ax.set_yticklabels(ylab); ax.invert_yaxis()
ax.set_xlim(0,100); ax.set_xlabel('Share of conditions (%)'); ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
h=[Patch(color=c,lw=0) for c in cols]
fig.legend(h,labs,loc='lower center',ncol=4,bbox_to_anchor=(0.56,0.0),handlelength=1.0,handleheight=0.8,columnspacing=0.8,handletextpad=0.35)
save(fig,os.path.join(OUT,'fig_selection'))
json.dump(shares,open(os.path.join(OUT,'fig_selection_values.json'),'w'),indent=0); print(shares)
