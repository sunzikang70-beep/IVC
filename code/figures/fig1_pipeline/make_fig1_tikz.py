import os, numpy as np, pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex
HERE=os.path.dirname(os.path.abspath(__file__))
D=os.environ.get('SAREF_FIG1_CASE',os.path.join(HERE,'..','..','..','outputs','fig1_pipeline_assets_20261105','case_5_3_225'))+os.sep
OUTTEX=os.environ.get('SAREF_FIG1_TEX',os.path.join(HERE,'..','..','figures','fig_pipeline.tex'))
img=os.environ.get('SAREF_FIG1_IMG','fig1_assets/')
order=['protocolF_se3_trust_p2p_s025','protocolF_se3_trust_p2p_s050','protocolF_se3_trust_p2p_s100',
       'protocolF_se3_trust_filtered_s025','protocolF_se3_trust_filtered_s050','protocolF_se3_trust_filtered_s100']
ccol=['9ECAE1','4292C6','08519C','DFC27D','BF812D','8C510A']
cp=pd.read_csv(D+'candidate_predictions.csv').set_index('action').loc[order]
vals=np.c_[cp.pred_delta_AR_VSD*100,cp.pred_delta_AR_MSSD*100,cp.pred_delta_AR_MSPD*100,cp.pred_add,cp.risk_score]
good=np.clip(np.c_[vals[:,0]/15,vals[:,1]/15,vals[:,2]/15,-vals[:,3]/1.2,(0.3-vals[:,4])/0.6],-1,1)
cm=LinearSegmentedColormap.from_list('g',['#B35806','#FDB863','#FFFFFF','#B2ABD2','#542788'])  # PuOr: harm orange, gain purple
u=(cp.u*100).to_numpy()
gates=np.c_[cp.gate_moving_valid&cp.gate_observation&cp.gate_domain,cp.gate_risk,cp.g_add,cp.g_comp,cp.g_floor,cp.g_u].astype(bool)
elig=gates.all(1); best=int(np.argmax(np.where(elig,u,-np.inf)))
def col(h): return '{rgb,255:red,%d;green,%d;blue,%d}'%tuple(int(h[i:i+2],16) for i in (0,2,4))
# feature families (counts from feature_schema.json, 191 names)
FAM=[('rendered depth',36,'0072B2'),('silhouette, boundary',32,'56B4E9'),('nearest neighbour',28,'009E73'),('joint support change',7,'F0E442'),
     ('bank registration scores',60,'CC79A7'),('motion, flags',10,'D55E00'),('target context',18,'7F7F7F')]  # Okabe-Ito categories
assert sum(f[1] for f in FAM)==191
L=[]; A=L.append
A(r'''\documentclass[tikz,border=2pt]{standalone}
\usepackage[T1]{fontenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb,graphicx,pifont,textcomp}
\usepackage[italic,LGRgreek]{mathastext}
\usetikzlibrary{arrows.meta,calc}
\definecolor{ink}{HTML}{1F2933}\definecolor{edge}{HTML}{9AA5B1}
\definecolor{tz}{HTML}{E69F00}\definecolor{gt}{HTML}{8426FF}\definecolor{sel}{HTML}{0072B2}
\definecolor{pA}{HTML}{2C7FB8}\definecolor{pF}{HTML}{8C510A}
\definecolor{okc}{HTML}{1E7B3A}\definecolor{noc}{HTML}{C0392B}
\begin{document}
\begin{tikzpicture}[x=1.075cm,y=1.05cm,font=\scriptsize,text=ink,
 arr/.style={-{Stealth[length=2mm,width=1.7mm]},line width=0.9pt,draw=ink!70},
 tarr/.style={-{Stealth[length=1.5mm,width=1.3mm]},line width=0.6pt,draw=ink!60},
 lab/.style={font=\scriptsize,align=center,inner sep=0.5pt},
 ttl/.style={font=\scriptsize\bfseries,align=center,inner sep=0.5pt},
 big/.style={font=\footnotesize\bfseries,align=center,inner sep=0.5pt},
 sm/.style={font=\scriptsize,align=center,inner sep=0.5pt},
 tile/.style={draw=edge,line width=0.4pt},
 blk/.style={rounded corners=2pt,line width=0.6pt,align=center,font=\scriptsize,inner sep=1.5pt}]
''')
# ===================== TOP ROW =====================
# panels
A(r'\fill[fill={rgb,255:red,236;green,243;blue,251},rounded corners=8pt] (2.55,5.2) rectangle (7.25,9.45);')
A(r'\fill[fill={rgb,255:red,251;green,242;blue,225},rounded corners=8pt] (7.65,5.2) rectangle (14.75,9.45);')
A(r'\node[big] at (4.9,9.24) {Candidate generation}; \node[big] at (10.0,9.24) {Observable feature extraction};')
# input
A(r'''\node[inner sep=0] (rgb) at (1.15,8.15) {\includegraphics[width=1.5cm]{%sA1_input_rgb.png}}; \draw[tile] (rgb.south west) rectangle (rgb.north east);
\node[inner sep=0] (dep) at (0.75,6.55) {\includegraphics[width=0.72cm]{%sA2_depth.png}}; \draw[tile] (dep.south west) rectangle (dep.north east);
\node[inner sep=0] (cad) at (1.55,6.55) {\includegraphics[width=0.72cm]{%sA4_cad.png}}; \draw[tile] (cad.south west) rectangle (cad.north east);
\node[lab] at (1.15,5.9) {RGB-D, mask $\Omega$,}; \node[lab] at (1.15,5.63) {CAD $\mathcal{M}$, pose $T_0$};
'''%(img,img,img))
# panel 1
A(r'''\node[inner sep=0] at (3.2,8.2) {\includegraphics[width=0.95cm]{%sA3a_cloud_all.png}};
\node[inner sep=0] at (3.2,6.45) {\includegraphics[width=0.95cm]{%sA3b_cloud_reliability.png}};
\node[sm] at (3.22,7.6) {$\mathcal{P}$, model at $T_0$}; \node[sm] at (3.22,5.84) {reliability-filtered};
\node[blk,draw=pA,fill=pA!18,minimum width=0.95cm,minimum height=0.48cm] (icp1) at (4.4,8.2) {P2P ICP};
\node[blk,draw=pF,fill=pF!16,minimum width=0.95cm,minimum height=0.48cm] (icp2) at (4.4,6.45) {P2P ICP};
\draw[tarr] (3.7,8.2) -- (icp1.west); \draw[tarr] (3.7,6.45) -- (icp2.west);
\draw[arr,rounded corners=3pt] (2.12,7.35) -- (2.4,7.35) |- (2.72,8.2); \draw[arr,rounded corners=3pt] (2.4,7.35) |- (2.72,6.45);
'''%(img,img))
# clip box with mini diagram
bx0,by0,bx1,by1=5.3,6.55,6.42,7.95
A(r'\fill[fill={rgb,255:red,253;green,234;blue,216},rounded corners=2pt,draw=tz!70,line width=0.6pt] (%.2f,%.2f) rectangle (%.2f,%.2f);'%(bx0,by0,bx1,by1))
ox,oy=5.45,6.7; radii=[0.24,0.48,0.96]
for r in radii:
    A(r'\draw[ink!40,line width=0.4pt] ([shift={(5:%.2f)}]%.2f,%.2f) arc (5:85:%.2f);'%(r,ox,oy,r))
for ang,Lf,c,cs in [(60,1.18,'pA',ccol[:3]),(24,1.1,'pF',ccol[3:])]:
    a=np.radians(ang); ex,ey=ox+Lf*np.cos(a),oy+Lf*np.sin(a)
    A(r'\draw[-{Stealth[length=1.2mm]},%s,line width=0.55pt,densely dashed] (%.2f,%.2f) -- (%.3f,%.3f);'%(c,ox,oy,ex,ey))
    for r,cc in zip(radii,cs):
        A(r'\fill[fill=%s,draw=white,line width=0.3pt] (%.3f,%.3f) circle (0.05);'%(col(cc),ox+r*np.cos(a),oy+r*np.sin(a)))
A(r'\fill[tz] (%.2f,%.2f) circle (0.055);'%(ox,oy))
A(r'''\node[ttl] at (5.86,8.14) {clip at caps};
\node[sm] at (5.86,6.38) {$s\in\{\tfrac14,\tfrac12,1\}$};
\draw[tarr,rounded corners=2pt] (icp1.east) -- (5.0,8.2) -- (5.0,7.55) -- (5.28,7.55);
\draw[tarr,rounded corners=2pt] (icp2.east) -- (5.0,6.45) -- (5.0,6.95) -- (5.28,6.95);''')
for i,cc in enumerate(ccol):
    y=7.7-0.2*i
    A(r'\fill[fill=%s,rounded corners=0.6pt] (6.7,%.3f) rectangle (7.06,%.3f);'%(col(cc),y-0.07,y+0.07))
A(r'''\node[sm] at (6.88,7.95) {$T_{1..6}$};
\draw[tarr] (6.44,7.2) -- (6.66,7.2);''')
# panel 2: maps
A(r'\node[ttl] at (8.78,8.78) {rendered at $T_i$};')
for f,x,y,t in [('A6_resid_sel',8.25,8.27,'residual'),('A7_signed',9.3,8.27,'signed'),('A8_silhouette',8.25,7.08,'silhouette'),('A9_nn_sel',9.3,7.08,'NN dist.')]:
    A(r'\node[inner sep=0] (n) at (%.2f,%.2f) {\includegraphics[width=0.62cm]{%s%s.png}}; \draw[tile] (n.south west) rectangle (n.north east); \node[sm] at (%.2f,%.2f) {%s};'%(x,y,img,f,x,y-0.44,t))
A(r'\draw[arr] (7.1,7.2) -- (7.86,7.2);')
# family bars
xs=[10.3,10.9,11.5,12.1,12.8,13.4,14.0]; yb0=6.9; hs=0.031
for (nm,n,c),x in zip(FAM,xs):
    A(r'\filldraw[fill=%s,draw=ink!45,line width=0.3pt] (%.2f,%.2f) rectangle (%.2f,%.3f); \node[sm] at (%.2f,%.2f) {%d};'%(col(c),x-0.14,yb0,x+0.14,yb0+n*hs,x,yb0-0.15,n))
A(r'''\draw[arr] (9.7,7.45) -- (10.08,7.45); \node[sm,anchor=south] at (9.89,7.5) {stats};''')
# skip connections
A(r'''\draw[tarr,rounded corners=3pt] (6.88,7.82) -- (6.88,9.0) -| (12.8,%.3f);
\draw[tarr,rounded corners=3pt] (12.8,9.0) -| (13.4,%.3f);
\node[sm,anchor=east] at (12.58,8.76) {bank scores:}; \node[sm,anchor=east] at (12.58,8.5) {value, $\Delta$, rank, gap};'''%(yb0+60*hs+0.02,yb0+10*hs+0.02))
A(r'''\draw[tarr,rounded corners=3pt] (1.15,8.92) -- (1.15,9.62) -| (14.0,%.3f);
\node[sm,anchor=south] at (7.6,9.64) {initializer and mask scores, depth-reliability statistics};'''%(yb0+18*hs+0.02))
# legend
leg=[(7.85,6.3),(7.85,6.03),(7.85,5.76),(7.85,5.49),(11.0,6.3),(11.0,6.03),(11.0,5.76)]
for (nm,n,c),(x,y) in zip(FAM,leg):
    A(r'\filldraw[fill=%s,draw=ink!45,line width=0.3pt] (%.2f,%.2f) rectangle (%.2f,%.2f); \node[sm,anchor=west] at (%.2f,%.2f) {%s (%d)};'%(col(c),x,y-0.07,x+0.3,y+0.07,x+0.36,y,nm,n))
A(r'\node[sm,anchor=west] at (11.0,5.49) {number of features per candidate};')
# output feature stack
segs=np.cumsum([0]+[f[1] for f in FAM])/191
for k in range(5,-1,-1):
    x0=15.08+0.08*k; y0=6.75+0.08*k; h=1.75
    for (nm,n,c),a0,a1 in zip(FAM,segs[:-1],segs[1:]):
        A(r'\fill[fill=%s] (%.3f,%.3f) rectangle (%.3f,%.3f);'%(col(c) if k==0 else col('E8EBEF'),x0,y0+a0*h,x0+0.26,y0+a1*h))
    A(r'\draw[white,line width=0.3pt] (%.3f,%.3f) rectangle (%.3f,%.3f);'%(x0,y0,x0+0.26,y0+h))
A(r'''\draw[arr] (14.3,7.5) -- (15.02,7.5);
\node[sm] at (15.45,6.5) {$X\in\mathbb{R}^{6\times191}$};
\node[lab] at (15.45,8.95) {candidate};\node[lab] at (15.45,8.72) {features};''')
# ===================== BOTTOM ROW =====================
boxes=[('Recall change','GBDT $\\times3\\to\\widehat{\\Delta a}_{ik}$',3.97,'E3F1DC','7FB069'),
       ('ADD-S change','GBDT $\\to\\widehat{\\Delta e}_i$',2.92,'DDE8F6','6F98CF'),
       ('ADD-S risk','classifier $\\to p_i^{+}$',1.87,'ECE4F5','9C84C6')]
for t,s,y,f,e in boxes:
    A(r'\fill[fill=%s,draw=%s,line width=0.6pt,rounded corners=3pt] (13.75,%.2f) rectangle (17.1,%.2f);'%(col(f),col(e),y-0.42,y+0.42))
    A(r'\node[big,anchor=west] at (13.88,%.2f) {%s}; \node[lab,anchor=west] at (13.88,%.2f) {%s};'%(y+0.15,t,y-0.17,s))
    for x in [16.28,16.64]:
        A(r'''\begin{scope}[shift={(%.2f,%.2f)},scale=0.1]
\draw[ink!55,line width=0.3pt] (0,0)--(-1,-1) (0,0)--(1,-1) (-1,-1)--(-1.5,-2) (-1,-1)--(-0.5,-2) (1,-1)--(1.5,-2) (1,-1)--(0.5,-2);
\foreach \p in {(0,0),(-1,-1),(1,-1),(-1.5,-2),(-0.5,-2),(0.5,-2),(1.5,-2)} \fill[fill=%s] \p circle (0.33);
\end{scope}'''%(x,y+0.12,col(e)))
    A(r'\node[font=\scriptsize] at (16.95,%.2f) {$\cdots$};'%(y-0.02))
    A(r'\draw[arr] (17.32,%.2f) -- (17.12,%.2f);'%(y,y))
A(r'''\draw[line width=0.9pt,draw=ink!70,rounded corners=4pt] (15.6,7.5) -- (17.32,7.5) -- (17.32,1.87);
\node[sm] at (15.43,1.24) {fitted on 22 T-LESS objects, then frozen};''')
# prediction matrix
mx0,cw,ytop,rh=11.2,0.38,3.42,0.25
rowy=[ytop-rh*(i+0.5) for i in range(6)]
for i,y in enumerate(rowy):
    A(r'\fill[fill=%s,rounded corners=0.5pt] (10.94,%.3f) rectangle (11.1,%.3f);'%(col(ccol[i]),y-0.08,y+0.08))
    for j in range(5):
        h=to_hex(cm((good[i,j]+1)/2))[1:]
        A(r'\fill[fill=%s,draw=white,line width=0.5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);'%(col(h),mx0+cw*j,y-rh/2,mx0+cw*(j+1),y+rh/2))
A(r'\node[sm] at (%.3f,3.6) {$\widehat{\Delta a}_{ik}$}; \node[sm] at (%.3f,3.6) {$\widehat{\Delta e}_i$}; \node[sm] at (%.3f,3.6) {$p_i^{+}$}; \node[sm] at (%.3f,1.6) {$(6,5)$};'%(mx0+1.5*cw,mx0+3.5*cw,mx0+4.5*cw,mx0+2.5*cw))
A(r'\shade[left color={rgb,255:red,179;green,88;blue,6},middle color=white,right color={rgb,255:red,84;green,39;blue,136}] (%.3f,1.24) rectangle (%.3f,1.34); \node[sm,anchor=east] at (%.3f,1.29) {harm}; \node[sm,anchor=west] at (%.3f,1.29) {gain};'%(mx0+0.45,mx0+1.45,mx0+0.42,mx0+1.48))
for t,s,y,f,e in boxes:
    A(r'\draw[tarr,rounded corners=2pt] (13.73,%.2f) -- (13.4,%.2f) -- (13.4,%.2f) -- (13.15,%.2f);'%(y,y,2.67+(y-2.92)*0.35,2.67+(y-2.92)*0.35))
# constraint box
A(r'''\fill[fill={rgb,255:red,252;green,233;blue,218},draw=tz!70,line width=0.6pt,rounded corners=3pt] (8.18,1.45) rectangle (10.68,3.85);
\node[big] at (9.43,3.6) {Constraints};
\node[sm] at (9.43,3.28) {moves, supported};
\node[sm] at (9.43,2.98) {$p_i^{+}\le0.3$,\ \ $\widehat{\Delta e}_i\le0$};
\node[sm] at (9.43,2.66) {$\min_k\widehat{\Delta a}_{ik}\ge-1$\,pp};
\node[sm] at (9.43,2.34) {$\widehat{\Delta\bar a}_i\ge-0.1$\,pp};
\node[sm] at (9.43,2.04) {$\widehat u_i>0$};
\node[sm] at (9.43,1.72) {(all must hold)};
\draw[arr] (10.9,2.67) -- (10.71,2.67);''')
# mask column
for i,y in enumerate(rowy):
    ok=elig[i]
    A(r'\fill[fill=%s,draw=white,line width=0.5pt] (7.62,%.3f) rectangle (7.94,%.3f); \node[font=\scriptsize,text=%s,inner sep=0] at (7.78,%.3f) {\ding{%s}};'%(col('D8DAEB' if ok else 'FEE0B6'),y-rh/2,y+rh/2,col('542788' if ok else 'B35806'),y,'51' if ok else '55'))
A(r'\node[lab] at (7.78,3.62) {$m_i$}; \draw[arr] (8.16,2.67) -- (7.98,2.67);')
# utility bars + one-hot
zero=6.15; sc=0.68/max(u)
for i,y in enumerate(rowy):
    fc=('0072B2' if i==best else '9ECAE1') if elig[i] else 'D0D0D0'
    A(r'\fill[fill=%s] (%.3f,%.3f) rectangle (%.3f,%.3f);'%(col(fc),zero,y-0.075,zero+u[i]*sc,y+0.075))
    on=(i==best)
    A(r'\fill[fill=%s,draw=%s,line width=0.4pt] (5.12,%.3f) rectangle (5.42,%.3f); \node[font=\scriptsize,text=%s,inner sep=0] at (5.27,%.3f) {%d};'%('sel' if on else 'white','sel!60' if on else 'edge',y-rh/2+0.01,y+rh/2-0.01,'white' if on else 'ink!45',y,1 if on else 0))
A(r'''\draw[ink!50,line width=0.4pt] (%.2f,%.3f) -- (%.2f,%.3f);
\node[lab] at (6.5,3.62) {$\widehat u_i$}; \node[lab] at (5.27,3.62) {argmax};
\draw[tarr] (7.58,2.67) -- (7.0,2.67);
\draw[tarr] (5.95,%.3f) -- (5.46,%.3f);
\node[sm] at (6.35,1.5) {$\widehat u_i=\widehat{\Delta\bar a}_i-0.005\,\widehat{\Delta e}_i/5$\,mm};
\node[sm] at (6.35,1.12) {$i^\ast=\arg\max_{m_i=1}\widehat u_i$};'''%(zero,rowy[0]+0.16,zero,rowy[-1]-0.16,rowy[best],rowy[best]))
# candidate poses down to selection
A(r'''\draw[tarr,dashed,rounded corners=3pt] (6.88,6.62) -- (6.88,4.3) -| (5.27,3.78);
\node[sm,anchor=west] at (6.95,4.85) {$T_1,\dots,T_6$};''')
# output
yb=rowy[best]
A(r'''\node[inner sep=0] (outp) at (2.4,%.3f) {\includegraphics[width=1.6cm]{%sA12_output.png}}; \draw[tile] (outp.south west) rectangle (outp.north east);
\node[ttl] at (2.4,3.82) {refined pose $\widehat T=T_{i^\ast}$};
\draw[arr] (5.08,%.3f) -- (3.24,%.3f); \node[lab,anchor=south] at (4.15,%.3f) {select};
\fill[gt,opacity=0.45] (1.72,1.42) rectangle (1.98,1.58); \node[sm,anchor=west] at (2.01,1.5) {ground truth (filled)};
\draw[tz,line width=1.1pt] (1.72,1.22)--(1.98,1.22); \node[sm,anchor=west] at (2.01,1.22) {input $T_0$};
\draw[sel,line width=1.1pt,densely dashed] (1.72,0.94)--(1.98,0.94); \node[sm,anchor=west] at (2.01,0.94) {output $\widehat T$};
\draw[tarr,dashed,rounded corners=3pt] (0.45,5.5) -- (0.45,%.3f) -- (1.56,%.3f);
\node[sm,rotate=90] at (0.3,4.0) {$T_0$ if no $m_i=1$};
'''%(yb,img,yb,yb,yb+0.03,yb,yb))
A(r'\end{tikzpicture}\end{document}')
open(OUTTEX,'w',encoding='utf-8').write('\n'.join(L)); print('wrote',OUTTEX)
