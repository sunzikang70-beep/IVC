import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"pdf.fonttype":42,"font.family":"DejaVu Sans","font.size":11})
INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8984"; GRID="#e6e5e0"
OURS="#2a78d6"; FIXED="#eb6834"; KEEP="#c2410c"
W,H=1328/100,531/100
fig=plt.figure(figsize=(W,H),dpi=200,facecolor="white")
# ---- left schematic
ax=fig.add_axes([0.0,0.0,0.47,1.0]); ax.set_xlim(0,47); ax.set_ylim(0,40); ax.axis("off")
ax.text(1.5,36.5,"Accept a local update only when it is predicted to help",fontsize=12.5,weight="bold",color=INK)
boxes=[(1.0,17,"External\npose $T_0$,\ndepth, mask,\nCAD model"),
       (12.6,17,"6 bounded\nupdates:\nP2P, filtered\nP2P x 3 caps"),
       (24.2,17,"Predicted\n$\\Delta$recall and\n$\\Delta$ADD-S\nper update"),
       (35.8,17,"Gates +\nmax utility")]
for x,y,t in boxes:
    ax.add_patch(FancyBboxPatch((x,y),9.4,13,boxstyle="round,pad=0.25,rounding_size=0.8",fc="#f3f6fb" if "Predicted" in t else "#f7f7f5",ec="#9aa3ad",lw=1.1))
    ax.text(x+4.7,y+6.5,t,ha="center",va="center",fontsize=9.3,color=INK,linespacing=1.35)
for x in [10.7,22.3,33.9]:
    ax.add_patch(FancyArrowPatch((x,23.5),(x+1.7,23.5),arrowstyle="-|>",mutation_scale=13,color="#52606d",lw=1.4))
# outputs
ax.add_patch(FancyArrowPatch((42.5,16.6),(42.5,9.2),arrowstyle="-|>",mutation_scale=13,color=OURS,lw=1.6))
ax.text(42.5,6.6,"accept best\ncandidate",ha="center",va="center",fontsize=9.4,color=INK)
ax.add_patch(FancyArrowPatch((37.5,16.6),(5.7,16.6),connectionstyle="arc3,rad=-0.3",arrowstyle="-|>",mutation_scale=13,color=KEEP,lw=1.4,ls="--"))
ax.text(21.6,6.2,"no candidate passes: keep $T_0$ unchanged",ha="center",fontsize=9.4,color=INK2)
# ---- right: three small panels, grouped by dataset (one measure each, no shared axis)
import numpy as np
specs=[("AR gain over input pose (pp)",[[1.12,4.25],[2.32,3.81]],"{:+.2f}",(0,5.2),["T-LESS","YCB-V"]),
       ("Accepted updates that\nworsen ADD-S (%)",[[25.9,8.4],[13.1,1.4]],"{:.1f}",(0,31),["T-LESS","YCB-V"]),
       ("Worst initialization\nstratum, AR change (pp)",[[-2.15],[0.72]],"{:+.2f}",(-2.9,1.2),["worst of 6 strata"])]
x0=0.50; w=0.145; gap=0.025
for i,(title,vals,fmt,lim,groups) in enumerate(specs):
    a=fig.add_axes([x0+i*(w+gap),0.24,w,0.54])
    n=len(groups); xs=np.arange(n); bw=0.32
    for j,(col,lab) in enumerate([(FIXED,"best fixed refinement"),(OURS,"SARef (ours)")]):
        bars=a.bar(xs+(j-0.5)*bw*1.25,vals[j],width=bw,color=col,edgecolor="white",linewidth=1.5,zorder=3,label=lab)
        for b,v in zip(bars,vals[j]):
            off=(lim[1]-lim[0])*0.03
            a.text(b.get_x()+b.get_width()/2, v+off if v>=0 else v-off, fmt.format(v), ha="center",
                   va="bottom" if v>=0 else "top", fontsize=8.0, color=INK, weight="bold")
    a.set_ylim(*lim); a.set_xlim(-0.6,n-0.4)
    a.axhline(0,color=MUTED,lw=0.9,zorder=2)
    a.grid(axis="y",color=GRID,lw=0.8,zorder=0)
    for sp in ["top","right","left","bottom"]: a.spines[sp].set_visible(False)
    a.tick_params(axis="y",colors=INK2,labelsize=8.5,length=0)
    a.set_xticks(xs); a.set_xticklabels(groups,fontsize=9.2,color=INK); a.tick_params(axis="x",length=0,pad=4)
    a.set_title(title,fontsize=9.6,color=INK,pad=8,loc="left")
    if i==0:
        leg=a.legend(loc="upper left",bbox_to_anchor=(0.0,-0.13),ncol=2,frameon=False,fontsize=9,handlelength=1.2,columnspacing=1.2)
fig.text(0.50,0.035,"Frozen T-LESS model. 8 unseen T-LESS objects (1,686 conditions), 21 YCB-V objects (24,144).",fontsize=8.2,color=INK2)
fig.savefig("graphical_abstract.png",dpi=200,facecolor="white")
fig.savefig("graphical_abstract.pdf",facecolor="white")
