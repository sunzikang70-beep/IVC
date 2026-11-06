import numpy as np, matplotlib, pandas as pd
matplotlib.use("Agg"); matplotlib.rcParams.update({"pdf.fonttype":42,"font.family":"DejaVu Sans"})
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from scipy import ndimage
import os
__file__=os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__),'common.py'),encoding='utf-8').read())
OUT=os.path.join(HERE,'..','..','figures','fig1_assets')+os.sep; os.makedirs(OUT,exist_ok=True)
y0,y1,x0,x1=240,470,323,553
mask=obs['mask'][y0:y1,x0:x1]; Dm=obs['depth_method'][y0:y1,x0:x1]*1000
P=obs['target_points']*np.array([1,-1,1])*1000; rho=obs['reliability']
lo,hi=np.percentile(P,[2,98],axis=0); ctr=(lo+hi)/2; r=(hi-lo).max()/2*1.05
def save(fig,name):
    fig.savefig(OUT+name+'.png',dpi=300,facecolor='white'); fig.savefig(OUT+name+'.pdf',facecolor='white'); plt.close(fig)
def ax3d():
    fig=plt.figure(figsize=(1.6,1.6)); ax=fig.add_axes([0,0,1,1],projection='3d')
    ax.view_init(elev=-38,azim=-62); ax.set_axis_off()
    ax.set_xlim(ctr[0]-r,ctr[0]+r); ax.set_ylim(ctr[1]-r,ctr[1]+r); ax.set_zlim(ctr[2]-r,ctr[2]+r)
    ax.set_box_aspect((1,1,1),zoom=1.35); return fig,ax
sc=lambda ax,p,c,s=0.5,**k: ax.scatter(p[:,0],p[:,1],p[:,2],c=c,s=s,linewidths=0,depthshade=False,**k)
R0,t0=POSES['protocolB_e2e_initial_pose']
Pm0=(R0@V.T).T+t0
# visible model points at T0 (render-based)
def visible(Rk,tk):
    Pm=(Rk@V.T).T+tk; uvw=(K@Pm.T).T; uv=np.round(uvw[:,:2]/uvw[:,2:3]-0.5).astype(int)
    zr=render(Rk,tk,crop=(y0,y1,x0,x1))
    inb=(uv[:,1]>=y0)&(uv[:,1]<y1)&(uv[:,0]>=x0)&(uv[:,0]<x1)
    vis=np.zeros(len(V),bool); idx=np.where(inb)[0]
    vis[idx]=(Pm[idx,2]<=zr[uv[idx,1]-y0,uv[idx,0]-x0]+1.5)&mask[uv[idx,1]-y0,uv[idx,0]-x0]
    return Pm,vis,zr
Pm,vis0,_=visible(R0,t0)
fig,ax=ax3d(); sc(ax,P,'#7A8796',0.45); sc(ax,Pm[vis0],'#E69F00',1.2); save(fig,'A3a_cloud_all')
keep=rho>0.35
fig,ax=ax3d(); sc(ax,P[keep],'#0072B2',0.45); sc(ax,P[~keep],'#D55E00',0.6); save(fig,'A3b_cloud_reliability')
# forward NN distance map in image space for T0 and selected
tree=cKDTree(P)
def nnmap(k):
    Rk,tk=POSES[k]; zr=render(Rk,tk,crop=(y0,y1,x0,x1))
    ys,xs=np.where(np.isfinite(zr)&mask)
    z=zr[ys,xs]; u=xs+x0+0.5; v=ys+y0+0.5
    X=np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z]
    d=tree.query(X)[0]; m=np.full(zr.shape,np.nan); m[ys,xs]=d; return m
for k,tag in [('protocolB_e2e_initial_pose','T0'),('protocolF_se3_trust_p2p_s100','sel')]:
    m=nnmap(k); fig=plt.figure(figsize=(1.6,1.6)); ax=fig.add_axes([0,0,1,1]); ax.set_axis_off()
    ax.imshow(np.ones_like(m),cmap='gray',vmin=0,vmax=1); ax.imshow(m,cmap='viridis_r',vmin=0,vmax=10)
    ax.contour(mask.astype(float),levels=[0.5],colors=['#888'],linewidths=0.5,linestyles='--'); save(fig,'A9_nn_'+tag)
    print(tag,'median NN %.2f'%np.nanmedian(m))
fig=plt.figure(figsize=(1.2,0.22)); cax=fig.add_axes([0.05,0.45,0.9,0.5])
cb=plt.colorbar(plt.cm.ScalarMappable(cmap='viridis_r',norm=plt.Normalize(0,10)),cax=cax,orientation='horizontal'); cb.set_ticks([0,5,10]); cb.ax.tick_params(labelsize=5,length=1.5,pad=1); cb.outline.set_linewidth(0.3); save(fig,'A9_cbar')
fig=plt.figure(figsize=(1.2,0.22)); cax=fig.add_axes([0.05,0.45,0.9,0.5])
cb=plt.colorbar(plt.cm.ScalarMappable(cmap='RdBu',norm=plt.Normalize(-15,15)),cax=cax,orientation='horizontal'); cb.set_ticks([-15,0,15]); cb.ax.tick_params(labelsize=5,length=1.5,pad=1); cb.outline.set_linewidth(0.3); save(fig,'A7_cbar')
# A11 gates with wider bar axis
order=['protocolF_se3_trust_p2p_s025','protocolF_se3_trust_p2p_s050','protocolF_se3_trust_p2p_s100','protocolF_se3_trust_filtered_s025','protocolF_se3_trust_filtered_s050','protocolF_se3_trust_filtered_s100']
cp=pd.read_csv(D+'candidate_predictions.csv').set_index('action').loc[order]
u=cp.u*100
gates=np.c_[cp.gate_moving_valid&cp.gate_observation&cp.gate_domain,cp.gate_risk,cp.g_add,cp.g_comp,cp.g_floor,cp.g_u].astype(bool)
fig=plt.figure(figsize=(2.7,1.45)); ax=fig.add_axes([0,0,0.6,1]); ax.set_axis_off()
ax.imshow(np.where(gates,1,0),cmap=LinearSegmentedColormap.from_list('gt',['#F6C9C4','#CDEBC5']),vmin=0,vmax=1,aspect='auto')
for i in range(6):
    for j in range(6): ax.text(j,i,'✓' if gates[i,j] else '✗',ha='center',va='center',fontsize=7.5,color='#1E6B34' if gates[i,j] else '#B3261E',fontweight='bold')
for i in range(7): ax.axhline(i-0.5,color='white',lw=1.2)
for j in range(7): ax.axvline(j-0.5,color='white',lw=1.2)
bx=fig.add_axes([0.63,0.0,0.37,1]); elig=gates.all(1); best=int(np.argmax(np.where(elig,u,-np.inf)))
cols=[('#0B4F9C' if i==best else ('#9CC3E6' if elig[i] else '#D9D9D9')) for i in range(6)]
bx.barh(np.arange(6),u,color=cols,height=0.62); bx.axvline(0,color='#555',lw=0.6); bx.set_ylim(5.5,-0.5); bx.set_xlim(-7,16)
for i in range(6): bx.text(u.iloc[i]+(0.5 if u.iloc[i]>=0 else -0.5),i,f"{u.iloc[i]:+.1f}",va='center',ha='left' if u.iloc[i]>=0 else 'right',fontsize=5.8,color='#111')
bx.set_axis_off(); save(fig,'A11_gates')
