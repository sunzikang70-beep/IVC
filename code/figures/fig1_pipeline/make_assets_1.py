import numpy as np, matplotlib, json, pandas as pd, time
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype":42,"font.family":"DejaVu Sans"})
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from scipy import ndimage
import os
__file__=os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__),'common.py'),encoding='utf-8').read())
OUT=os.path.join(HERE,'..','..','figures','fig1_assets')+os.sep; os.makedirs(OUT,exist_ok=True); import os; os.makedirs(OUT,exist_ok=True)
C=dict(T0='#E69F00',gt='#8426FF',sel='#0072B2',mask='#FFFFFF',
       p2p=['#9ECAE1','#4292C6','#08519C'],fil=['#DFC27D','#BF812D','#8C510A'])
order=['protocolF_se3_trust_p2p_s025','protocolF_se3_trust_p2p_s050','protocolF_se3_trust_p2p_s100',
       'protocolF_se3_trust_filtered_s025','protocolF_se3_trust_filtered_s050','protocolF_se3_trust_filtered_s100']
cand_col=dict(zip(order,C['p2p']+C['fil']))
y0,y1,x0,x1=240,470,323,553
crop=(y0,y1,x0,x1)
rgb=obs['rgb'][y0:y1,x0:x1]; mask=obs['mask'][y0:y1,x0:x1]; Dm=obs['depth_method'][y0:y1,x0:x1]*1000; patch=obs['patch'][y0:y1,x0:x1]
t=time.time()
R={}
for k in ['protocolB_e2e_initial_pose',*order]: R[k]=render(*POSES[k],crop=crop)
R['gt']=render(*GT,crop=crop)
print('render s',time.time()-t)
def edge(m,thick=1):
    e=m&~ndimage.binary_erosion(m,iterations=thick); return e
def save(fig,name,dpi=300):
    fig.savefig(OUT+name+'.png',dpi=dpi,transparent=False,facecolor='white'); fig.savefig(OUT+name+'.pdf',facecolor='white'); plt.close(fig)
def imfig(w=1.6,h=1.6):
    fig=plt.figure(figsize=(w,h)); ax=fig.add_axes([0,0,1,1]); ax.set_axis_off(); return fig,ax
def overlay_contour(ax,m,color,lw=1.0,ls='-'):
    ax.contour(m.astype(float),levels=[0.5],colors=[color],linewidths=lw,linestyles=ls)
light=lambda im,a=0.55:(im*a+255*(1-a)).astype(np.uint8)
# A1 input RGB with mask and T0
fig,ax=imfig(); ax.imshow(light(rgb,0.8)); overlay_contour(ax,mask,'white',1.6); overlay_contour(ax,mask,'#222222',0.6,'--')
overlay_contour(ax,np.isfinite(R['protocolB_e2e_initial_pose']),C['T0'],1.3); save(fig,'A1_input_rgb')
# A2 depth with outlier patch
fig,ax=imfig(); d=np.where(Dm>0,Dm,np.nan); vmin,vmax=np.nanpercentile(d[mask],[1,99])
ax.imshow(np.where(Dm>0,Dm,np.nan),cmap='viridis',vmin=vmin-40,vmax=vmax+40)
overlay_contour(ax,mask,'white',1.4); overlay_contour(ax,patch,'#000000',1.2,'--'); save(fig,'A2_depth')
# 3D clouds
P=obs['target_points']*np.array([1,-1,1])*1000  # bop frame mm
rho=obs['reliability']
def cloud(ax,pts,c,cmap=None,vmin=None,vmax=None,s=0.6):
    ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=c,cmap=cmap,vmin=vmin,vmax=vmax,s=s,linewidths=0,depthshade=False)
    ax.view_init(elev=-62,azim=-90); ax.set_axis_off()
    ctr=P.mean(0); r=np.ptp(P,axis=0).max()/2*0.95
    ax.set_xlim(ctr[0]-r,ctr[0]+r); ax.set_ylim(ctr[1]-r,ctr[1]+r); ax.set_zlim(ctr[2]-r,ctr[2]+r)
for name,colors in [('A3a_cloud_all',None),('A3b_cloud_reliability','rho')]:
    fig=plt.figure(figsize=(1.6,1.6)); ax=fig.add_axes([-0.15,-0.15,1.3,1.3],projection='3d')
    if colors is None: cloud(ax,P,'#5B6B7F')
    else:
        keep=rho>0.35
        cloud(ax,P[~keep],'#F2A0A0',s=0.6); cloud(ax,P[keep],rho[keep],cmap='viridis',vmin=0.35,vmax=1.0)
    save(fig,name)
print('removed by filter',float((rho<=0.35).mean()))
# A4 CAD shaded (render GT pose full shading in crop)
Rg,tg=GT; dsh,sh=render(Rg,tg,crop=crop,shade=True)
img=np.where(np.isfinite(dsh),0.35+0.6*sh,1.0)
fig,ax=imfig(); ax.imshow(img,cmap='gray',vmin=0,vmax=1); save(fig,'A4_cad')
# A5 zoomed contours of 7 poses
sil={k:np.isfinite(v) for k,v in R.items()}
diff=(sil['protocolB_e2e_initial_pose']^sil['protocolF_se3_trust_p2p_s100']).astype(float)
dens=ndimage.uniform_filter(diff,40); cy,cx=np.unravel_index(np.argmax(dens),dens.shape)
hz=26; zy0,zy1,zx0,zx1=max(0,cy-hz),cy+hz,max(0,cx-hz),cx+hz
up=8
def zoom(a): return np.kron(a,np.ones((up,up)))
fig,ax=imfig(); ax.imshow(np.kron(light(rgb[zy0:zy1,zx0:zx1],0.6),np.ones((up,up,1))).astype(np.uint8))
overlay_contour(ax,zoom(sil['protocolB_e2e_initial_pose'][zy0:zy1,zx0:zx1]),C['T0'],1.6)
for k in order: overlay_contour(ax,zoom(sil[k][zy0:zy1,zx0:zx1]),cand_col[k],1.2)
save(fig,'A5_candidates_zoom')
fig,ax=imfig(); ax.imshow(light(rgb,0.7)); ax.add_patch(plt.Rectangle((zx0,zy0),zx1-zx0,zy1-zy0,fill=False,ec='#222',lw=1.0))
overlay_contour(ax,sil['protocolB_e2e_initial_pose'],C['T0'],1.0); save(fig,'A5b_zoom_locator')
json.dump(dict(zoom=[int(zy0),int(zy1),int(zx0),int(zx1)]),open(OUT+'zoom.json','w'))
# A6 residual maps |Dr - D| for T0 and selected
def resid(k):
    r=R[k]; v=np.isfinite(r)&mask&(Dm>0); out=np.full(r.shape,np.nan); out[v]=np.abs(r[v]-Dm[v]); return out
for k,tag in [('protocolB_e2e_initial_pose','T0'),('protocolF_se3_trust_p2p_s100','sel')]:
    fig,ax=imfig(); ax.imshow(np.full(Dm.shape,1.0),cmap='gray',vmin=0,vmax=1)
    im=ax.imshow(resid(k),cmap='magma_r',vmin=0,vmax=20); overlay_contour(ax,mask,'#666666',0.5,'--'); save(fig,'A6_resid_'+tag)
# colorbar asset
fig=plt.figure(figsize=(1.2,0.22)); cax=fig.add_axes([0.05,0.45,0.9,0.5]); cb=plt.colorbar(plt.cm.ScalarMappable(cmap='magma_r',norm=plt.Normalize(0,20)),cax=cax,orientation='horizontal')
cb.set_ticks([0,10,20]); cb.ax.tick_params(labelsize=5,length=1.5,pad=1); cb.outline.set_linewidth(0.3); save(fig,'A6_cbar')
r0=resid('protocolB_e2e_initial_pose'); r3=resid('protocolF_se3_trust_p2p_s100')
print('median resid T0 %.2f sel %.2f; p95 %.2f %.2f'%(np.nanmedian(r0),np.nanmedian(r3),np.nanpercentile(r0,95),np.nanpercentile(r3,95)))
# A7 signed free-space map for selected: observed minus rendered
k='protocolF_se3_trust_p2p_s100'; r=R[k]; v=np.isfinite(r)&(Dm>0)
s=np.full(r.shape,np.nan); s[v]=Dm[v]-r[v]
fig,ax=imfig(); ax.imshow(np.full(Dm.shape,1.0),cmap='gray',vmin=0,vmax=1); ax.imshow(np.clip(s,-15,15),cmap='RdBu',vmin=-15,vmax=15)
overlay_contour(ax,mask,'#666666',0.5,'--'); save(fig,'A7_signed')
print('front-violation frac (>10mm) sel %.3f'%np.nanmean(s[np.isfinite(s)]>10))
# A8 silhouette agreement for selected
sr=sil[k]; tp=sr&mask; ro=sr&~mask; mo=mask&~sr
img=np.ones(mask.shape+(3,)); img[tp]=[0.78,0.78,0.78]; img[ro]=[0.89,0.30,0.30]; img[mo]=[0.27,0.51,0.80]
fig,ax=imfig(); ax.imshow(img); save(fig,'A8_silhouette')
print('silhouette IoU sel %.3f, T0 %.3f'%((sr&mask).sum()/(sr|mask).sum(), (sil['protocolB_e2e_initial_pose']&mask).sum()/(sil['protocolB_e2e_initial_pose']|mask).sum()))
# A9 visible model points coloured by NN distance (selected pose)
Rk,tk=POSES[k]; Pm=(Rk@V.T).T+tk; uvw=(K@Pm.T).T; uv=np.round(uvw[:,:2]/uvw[:,2:3]-0.5).astype(int)
inb=(uv[:,1]>=y0)&(uv[:,1]<y1)&(uv[:,0]>=x0)&(uv[:,0]<x1)
vis=np.zeros(len(V),bool); zr=R[k]
vis[inb]=Pm[inb,2]<=zr[uv[inb,1]-y0,uv[inb,0]-x0]+1.5
vis&=np.r_[[mask[uv[i,1]-y0,uv[i,0]-x0] if inb[i] else False for i in range(len(V))]]
dist=cKDTree(P).query(Pm[vis])[0]
fig=plt.figure(figsize=(1.6,1.6)); ax=fig.add_axes([-0.15,-0.15,1.3,1.3],projection='3d')
cloud(ax,P,'#D9D9D9',s=0.4); cloud(ax,Pm[vis],np.clip(dist,0,8),cmap='plasma',vmin=0,vmax=8,s=1.5); save(fig,'A9_nn')
print('visible model pts',int(vis.sum()),'median NN %.2f'%np.median(dist))
# A10 prediction heatmap and A11 gate matrix
cp=pd.read_csv(D+'candidate_predictions.csv').set_index('action').loc[order]
vals=np.c_[cp.pred_delta_AR_VSD*100,cp.pred_delta_AR_MSSD*100,cp.pred_delta_AR_MSPD*100,cp.pred_add,cp.risk_score]
good=np.c_[vals[:,0]/15,vals[:,1]/15,vals[:,2]/15,-vals[:,3]/1.5,(0.3-vals[:,4])/0.5]
fig=plt.figure(figsize=(2.0,1.45)); ax=fig.add_axes([0,0,1,1]); ax.set_axis_off()
cm=LinearSegmentedColormap.from_list('g',['#D7301F','#FDF0E6','#FFFFFF','#E5F5E0','#238B45'])
ax.imshow(np.clip(good,-1,1),cmap=cm,vmin=-1,vmax=1,aspect='auto')
for i in range(6):
    for j in range(5):
        txt=f"{vals[i,j]:+.1f}" if j<4 else f"{vals[i,j]:.2f}"
        ax.text(j,i,txt,ha='center',va='center',fontsize=6.2,color='#111')
for i in range(7): ax.axhline(i-0.5,color='white',lw=1.2)
for j in range(6): ax.axvline(j-0.5,color='white',lw=1.2)
save(fig,'A10_predictions')
u=cp.u*100
gates=np.c_[cp.gate_moving_valid&cp.gate_observation&cp.gate_domain,cp.gate_risk,cp.g_add,cp.g_comp,cp.g_floor,cp.g_u].astype(bool)
fig=plt.figure(figsize=(2.6,1.45)); ax=fig.add_axes([0,0,0.62,1]); ax.set_axis_off()
ax.imshow(np.where(gates,1,0),cmap=LinearSegmentedColormap.from_list('gt',['#F6C9C4','#CDEBC5']),vmin=0,vmax=1,aspect='auto')
for i in range(6):
    for j in range(6): ax.text(j,i,'✓' if gates[i,j] else '✗',ha='center',va='center',fontsize=7.5,color='#1E6B34' if gates[i,j] else '#B3261E',fontweight='bold')
for i in range(7): ax.axhline(i-0.5,color='white',lw=1.2)
for j in range(7): ax.axvline(j-0.5,color='white',lw=1.2)
bx=fig.add_axes([0.66,0.0,0.34,1]); elig=gates.all(1); best=int(np.argmax(np.where(elig,u,-np.inf)))
cols=[('#0B4F9C' if i==best else ('#9CC3E6' if elig[i] else '#D9D9D9')) for i in range(6)]
bx.barh(np.arange(6),u,color=cols,height=0.62); bx.axvline(0,color='#555',lw=0.6); bx.set_ylim(5.5,-0.5); bx.set_xlim(-6,12)
for i in range(6): bx.text(u.iloc[i]+(0.4 if u.iloc[i]>=0 else -0.4),i,f"{u.iloc[i]:+.1f}",va='center',ha='left' if u.iloc[i]>=0 else 'right',fontsize=5.8,color='#111')
bx.set_axis_off(); save(fig,'A11_gates')
print('utility pp',u.round(2).tolist(),'best',order[best])
# A12 output overlay
fig,ax=imfig(); ax.imshow(light(rgb,0.8))
from matplotlib.colors import to_rgba
gfill=np.zeros(sil['gt'].shape+(4,)); gfill[sil['gt']]=to_rgba(C['gt'],0.30); ax.imshow(gfill)
overlay_contour(ax,sil['protocolB_e2e_initial_pose'],C['T0'],1.3); overlay_contour(ax,sil[k],C['sel'],1.3,'--')
save(fig,'A12_output')
