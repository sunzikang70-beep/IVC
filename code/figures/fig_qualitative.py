import sys, os, json, numpy as np; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
OUT=os.environ.get("FIGOUT","/home/claude/newfigs/out")
from PIL import Image, ImageDraw
Q=os.environ.get("QDATA","/mnt/user-data/uploads/codex_paper_workspace/outputs/a_main_method_qualitative_20261105/")
IMG=os.environ.get("QIMG","/home/claude/qualA/tless/")
cases=json.load(open(Q+"qualitative_cases.json"))
def ply(path):
    b=open(path,"rb").read(); h=b.index(b"end_header\n")+len(b"end_header\n"); hdr=b[:h].decode()
    nv=int(hdr.split("element vertex ")[1].split()[0]); nf=int(hdr.split("element face ")[1].split()[0])
    props=[l.split()[-1] for l in hdr.splitlines() if l.startswith("property") and "list" not in l]
    vstride=4*len(props)
    v=np.frombuffer(b[h:h+nv*vstride],dtype="<f4").reshape(nv,len(props))[:,:3].astype(float)
    off=h+nv*vstride; f=np.empty((nf,3),int)
    for i in range(nf):
        n=b[off]; f[i]=np.frombuffer(b[off+1:off+1+4*n],dtype="<i4")[:3]; off+=1+4*n
    return v,f
def silhouette(v,f,R,t,K,shape):
    P=(np.array(R).reshape(3,3)@v.T).T+np.array(t).reshape(1,3); K=np.array(K).reshape(3,3)
    uv=(K@P.T).T; uv=uv[:,:2]/uv[:,2:3]
    im=Image.new("L",(shape[1],shape[0]),0); d=ImageDraw.Draw(im)
    for tri in f: d.polygon([tuple(uv[i]) for i in tri],fill=255)
    m=np.array(im)>0
    er=m.copy()
    for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]: er&=np.roll(np.roll(m,dy,0),dx,1)
    e=m&~er
    e2=e.copy()
    for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]: e2|=np.roll(np.roll(e,dy,0),dx,1)
    return m, e2
CL={"gt":C["gt"],"initial":C["t0"],"method":C["sar"],"fixed":C["fix"]}
titles={"success":"Accepted update","correct_rejection":"Rejected update (input kept)","wrong_acceptance":"Accepted update with AR loss"}
order=["success","correct_rejection","wrong_acceptance"]
fig,axs=plt.subplots(2,3,figsize=(W2,136*MM)); fig.subplots_adjust(left=0.01,right=0.99,top=0.958,bottom=0.135,wspace=0.06,hspace=0.36)
for col,cat in enumerate(order):
    cs=[c for c in cases if c["case"]==cat]
    for row,c in enumerate(cs):
        ax=axs[row,col]
        img=np.array(Image.open(IMG+c["rgb"]).convert("RGB")); H,W=img.shape[:2]
        v,f=ply(Q+f"models_copy/obj_{c['object_id']:06d}.ply")
        layers=[("gt",c["gt"]),("initial",c["initial"]),("fixed" if cat=="correct_rejection" else "method",c["fixed"] if cat=="correct_rejection" else c["method"])]
        masks={}
        for k,p in layers: masks[k]=silhouette(v,f,p["R"],p["t"],c["K"],(H,W))
        ys,xs=np.where(masks["gt"][0]); cy,cx=(ys.min()+ys.max())/2,(xs.min()+xs.max())/2
        half=max(ys.max()-ys.min(),xs.max()-xs.min())*0.62
        y0,y1=int(max(0,cy-half)),int(min(H,cy+half)); x0,x1=int(max(0,cx-half)),int(min(W,cx+half))
        crop=(img[y0:y1,x0:x1]*0.55+255*0.45).astype(np.uint8)
        ax.imshow(crop,interpolation="lanczos")
        from matplotlib.colors import to_rgba
        g=np.zeros(masks["gt"][0][y0:y1,x0:x1].shape+(4,)); g[masks["gt"][0][y0:y1,x0:x1]]=to_rgba(CL["gt"],0.30); ax.imshow(g,interpolation="nearest")
        for k,(m,edge) in masks.items():
            if k=="gt": continue
            yy,xx=np.where(edge[y0:y1,x0:x1])
            ax.scatter(xx,yy,s=1.1,c=CL[k],linewidths=0,rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#C3C8CF"); sp.set_linewidth(0.5)
        if cat=="correct_rejection":
            txt=f"Fixed refinement: AR {c['AR_init']:.2f}→{c['AR_fixed']:.2f}\nSARef kept AR {c['AR_init']:.2f}"
        else:
            txt=f"AR {c['AR_init']:.2f}→{c['AR_method']:.2f}\nADD-S {c['initial']['ADD_S_mm']:.1f}→{c['method']['ADD_S_mm']:.1f} mm"
        ax.set_xlabel(f"obj {c['object_id']}, {c['init_regime']} init\n"+txt,fontsize=7,labelpad=2)
        if row==0: ax.set_title("("+"abc"[col]+") "+titles[cat],fontsize=7.5,fontweight="bold",loc="left")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
h=[Patch(facecolor=CL["gt"],alpha=0.30,lw=0)]+[Line2D([],[],color=CL[k],lw=2) for k in ["initial","method","fixed"]]
fig.legend(h,["Ground truth (filled)","Input pose","SARef selected pose","Fixed refinement (filtered P2P, cap 1)"],loc="lower center",ncol=4,bbox_to_anchor=(0.5,0.0),columnspacing=1.6)
save(fig,os.path.join(OUT,"fig_qualitative"))
print([ (c['case'],c['object_id'],round(c['AR_init'],3),round(c['AR_method'],3)) for c in cases])
