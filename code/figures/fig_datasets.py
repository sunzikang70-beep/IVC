"""Datasets and input conditions: test scenes, object galleries (T-LESS split), depth outliers, initial errors."""
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from figstyle import *; from common_data import *
import numpy as np, json
from PIL import Image
from matplotlib.patches import Rectangle
OUT=os.environ.get('FIGOUT','/home/claude/newfigs/out')
CASE=os.environ.get('FIGCASE','/mnt/user-data/uploads/codex_paper_workspace/outputs/fig1_pipeline_assets_20261105/case_5_3_225/case_observation.npz')
UNSEEN=[1,5,8,13,18,21,22,27]
FW,FH=190,139
fig=plt.figure(figsize=(FW*MM,FH*MM))
def ax_mm(x,y,w,h): return fig.add_axes([x/FW,y/FH,w/FW,h/FH])
def img_ax(x,y,w,h,im,edge='#C3C8CF',lw=0.5):
    a=ax_mm(x,y,w,h); a.imshow(im,interpolation='lanczos'); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(True); s.set_color(edge); s.set_linewidth(lw)
    return a
def title(x,y,s): fig.text(x/FW,y/FH,s,fontsize=7.5,fontweight='bold',ha='left',va='bottom')
# (a)/(c) scenes
title(0,134.5,'(a) T-LESS test scene'); img_ax(0,93.5,54,40.5,Image.open(IN+'/scenes/tless_000003_000225.png'))
title(0,88.5,'(c) YCB-V test scene'); img_ax(0,47.5,54,40.5,Image.open(IN+'/scenes/ycbv_000050_001874.png'))
# (b) T-LESS gallery
title(58,134.5,'(b) T-LESS objects: 22 development objects (grey) and 8 unseen objects (blue)')
cs,gap=13.0,0.22
for o in range(1,31):
    r,c=(o-1)//10,(o-1)%10; x=58+c*(cs+gap); y=93.5+(2-r)*(cs+gap)
    un=o in UNSEEN
    a=img_ax(x,y,cs,cs,Image.open(IN+f'/crops_tless/obj_{o:02d}.png'),edge=C['sar'] if un else '#C3C8CF',lw=1.1 if un else 0.5)
    a.text(0.04,0.96,str(o),transform=a.transAxes,ha='left',va='top',fontsize=7,color='white',fontweight='bold' if un else 'normal',
           bbox=dict(boxstyle='square,pad=0.15',fc=C['sar'] if un else '#7B8794',ec='none'))
# (d) YCB-V gallery
title(58,88.5,'(d) YCB-V objects (21, all used for external evaluation)')
for o in range(1,22):
    r,c=(o-1)//7,(o-1)%7; x=58+c*(cs+gap); y=47.5+(2-r)*(cs+gap)
    a=img_ax(x,y,cs,cs,Image.open(IN+f'/crops_ycbv/obj_{o:02d}.png'))
    a.text(0.04,0.96,str(o),transform=a.transAxes,ha='left',va='top',fontsize=7,color='white',bbox=dict(boxstyle='square,pad=0.15',fc='#7B8794',ec='none'))
# (e) depth conditions
z=np.load(CASE); y0,y1,x0,x1=240,470,323,553
dn=z['depth_native'][y0:y1,x0:x1]*1000; dm=z['depth_method'][y0:y1,x0:x1]*1000; patch=z['patch'][y0:y1,x0:x1]>0
v=dn[dn>0]; lo,hi=np.percentile(v,2)-30,np.percentile(v,98)+30
title(154,88.5,'(e) Depth input')
for k,(dd,lab) in enumerate([(dn,'native'),(dm,'30% outliers')]):
    a=ax_mm(154,47.5+(1-k)*20.6,19.6,19.6)
    im=a.imshow(np.where(dd>0,dd,np.nan),cmap='viridis',vmin=lo,vmax=hi,interpolation='nearest'); a.set_facecolor('white')
    if k==1: a.contour(patch.astype(float),levels=[0.5],colors=['black'],linewidths=0.6,linestyles='--')
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(True); s.set_color('#C3C8CF'); s.set_linewidth(0.5)
    a.text(0.03,0.04,lab,transform=a.transAxes,fontsize=7,va='bottom',ha='left',bbox=dict(boxstyle='square,pad=0.15',fc='white',ec='none',alpha=0.85))
cax=ax_mm(175.2,47.5,1.6,40.2); cb=fig.colorbar(im,cax=cax); cb.outline.set_linewidth(0.4); cb.ax.tick_params(length=1.8,width=0.5,labelsize=7)
cb.set_label('depth (mm)',fontsize=7,labelpad=1)
# (f) initial error distributions
title(0,36,'(f) Initial ADD-S error of the input pose by initialization regime')
stats={}
for k,(name,dn_) in enumerate([('tless','T-LESS (8 unseen objects)'),('ycbv','YCB-V (21 objects)')]):
    d=load(name); a=ax_mm(14+k*96,9.5,80,22.5)
    data=[np.log10(np.clip(d.loc[d.init_regime.eq(r),'e0'].to_numpy()*100,0.1,None)) for r in ['native','mild','hard']]
    vp=a.violinplot(data,positions=[2,1,0],orientation='horizontal',widths=0.8,showextrema=False)
    for b,c in zip(vp['bodies'],['#9AA5B1','#6F8FB8','#2E5C99']): b.set_facecolor(c); b.set_alpha(0.55); b.set_edgecolor('none')
    for p,dd in zip([2,1,0],data):
        q=np.percentile(dd,[25,50,75]); a.plot(q[[0,2]],[p,p],color=C['ink'],lw=1.6,solid_capstyle='butt'); a.plot(q[1],p,'o',ms=3,color='white',mec=C['ink'],mew=0.7)
    stats[name]={r:[round(float(x),4) for x in np.percentile(d.loc[d.init_regime.eq(r),'e0'],[25,50,75])] for r in ['native','mild','hard']}
    stats[name]['ge_1d']=round(float((d.e0>=1).mean()),4)
    for xv in [1,2]: a.axvline(xv,color=C['ink2'],lw=0.6,ls=':')
    a.set_yticks([2,1,0]); a.set_yticklabels(['native','mild','hard']); a.tick_params(axis='y',length=0)
    a.set_xlim(-0.2,3.3); a.set_xticks([0,1,2,3]); a.set_xticklabels(['1','10','100','1000'])
    a.set_xlabel('% of object diameter (log scale)',labelpad=1.5); a.spines['left'].set_visible(False)
    a.text(0.0,1.02,dn_,transform=a.transAxes,fontsize=7,ha='left',va='bottom')
save(fig,os.path.join(OUT,'fig_datasets'))
json.dump(stats,open(os.path.join(OUT,'fig_datasets_values.json'),'w'),indent=0); print(stats)
