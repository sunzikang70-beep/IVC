import numpy as np, json
import os
HERE=os.path.dirname(os.path.abspath(__file__))
# case folder written by outputs/fig1_pipeline_assets_20261105/source/extract_case.py (hash-verified replay)
D=os.environ.get('SAREF_FIG1_CASE',os.path.join(HERE,'..','..','..','outputs','fig1_pipeline_assets_20261105','case_5_3_225'))+os.sep
obs=np.load(D+'case_observation.npz'); meta=json.load(open(D+'case_meta.json'))
FLIP=np.diag([1.,-1.,1.])
def bop_pose(T):
    T=np.array(T); R=FLIP@T[:3,:3]; t=FLIP@T[:3,3]; return R,t   # mm
def ply(path):
    b=open(path,"rb").read(); h=b.index(b"end_header\n")+len(b"end_header\n"); hdr=b[:h].decode()
    nv=int(hdr.split("element vertex ")[1].split()[0]); nf=int(hdr.split("element face ")[1].split()[0])
    props=[l.split() for l in hdr.splitlines() if l.startswith("property") and "list" not in l]
    assert all(p[1]=='float' for p in props), props
    v=np.frombuffer(b[h:h+nv*4*len(props)],dtype="<f4").reshape(nv,len(props))[:,:3].astype(float)
    off=h+nv*4*len(props); f=np.empty((nf,3),int)
    for i in range(nf):
        n=b[off]; f[i]=np.frombuffer(b[off+1:off+1+4*n],dtype="<i4")[:3]; off+=1+4*n
    return v,f
V,F=ply(D+'obj_000005.ply')
K=obs['K']; H,W=obs['mask'].shape
POSES={k:bop_pose(v['T']) for k,v in meta['poses'].items()}
GT=(np.array(meta['gt']['R']).reshape(3,3),np.array(meta['gt']['t']))
NAMES={'protocolB_e2e_initial_pose':'T0','protocolF_se3_trust_p2p_s025':'P2P 0.25','protocolF_se3_trust_p2p_s050':'P2P 0.5','protocolF_se3_trust_p2p_s100':'P2P 1',
       'protocolF_se3_trust_filtered_s025':'Filt. 0.25','protocolF_se3_trust_filtered_s050':'Filt. 0.5','protocolF_se3_trust_filtered_s100':'Filt. 1'}
def render(R,t,crop=None,shade=False):
    """z-buffer rasterisation in pixel space; returns depth (mm, inf=empty) and optional shading"""
    P=(R@V.T).T+t; uvw=(K@P.T).T; uv=uvw[:,:2]/uvw[:,2:3]; z=P[:,2]
    y0,y1,x0,x1=crop if crop else (0,H,0,W)
    depth=np.full((y1-y0,x1-x0),np.inf); sh=np.zeros_like(depth)
    n=np.cross(P[F[:,1]]-P[F[:,0]],P[F[:,2]]-P[F[:,0]]); n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-12
    light=np.array([0.3,-0.5,-0.8]); light/=np.linalg.norm(light)
    for fi,(a,b,c) in enumerate(F):
        pa,pb,pc=uv[a],uv[b],uv[c]
        xmin=int(np.floor(min(pa[0],pb[0],pc[0]))); xmax=int(np.ceil(max(pa[0],pb[0],pc[0])))
        ymin=int(np.floor(min(pa[1],pb[1],pc[1]))); ymax=int(np.ceil(max(pa[1],pb[1],pc[1])))
        xmin=max(xmin,x0); ymin=max(ymin,y0); xmax=min(xmax,x1-1); ymax=min(ymax,y1-1)
        if xmin>xmax or ymin>ymax: continue
        xs,ys=np.meshgrid(np.arange(xmin,xmax+1)+0.5,np.arange(ymin,ymax+1)+0.5)
        d=(pb[1]-pc[1])*(pa[0]-pc[0])+(pc[0]-pb[0])*(pa[1]-pc[1])
        if abs(d)<1e-12: continue
        l1=((pb[1]-pc[1])*(xs-pc[0])+(pc[0]-pb[0])*(ys-pc[1]))/d
        l2=((pc[1]-pa[1])*(xs-pc[0])+(pa[0]-pc[0])*(ys-pc[1]))/d
        l3=1-l1-l2
        ins=(l1>=0)&(l2>=0)&(l3>=0)
        if not ins.any(): continue
        zz=l1*z[a]+l2*z[b]+l3*z[c]
        yy=(ys[ins]-0.5).astype(int)-y0; xx=(xs[ins]-0.5).astype(int)-x0
        cur=depth[yy,xx]; m=zz[ins]<cur
        depth[yy[m],xx[m]]=zz[ins][m]
        if shade: sh[yy[m],xx[m]]=abs(n[fi]@light)
    return (depth,sh) if shade else depth
