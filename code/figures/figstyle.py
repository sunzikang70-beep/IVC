"""Shared style for all matplotlib figures: Arial-metric sans (Liberation Sans), 7 pt lettering,
Elsevier widths (90 / 140 / 190 mm)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
MM=1/25.4
W1,W15,W2=90*MM,140*MM,190*MM
C=dict(sar="#0072B2",fix="#1A1A1A",veto="#9AA0A6",t0="#E69F00",gt="#8426FF",grid="#E6E8EB",ink="#1F2933",ink2="#52606D",
       cand=["#9ECAE1","#4292C6","#08519C","#DFC27D","#BF812D","#8C510A"],keep="#D5D9DE")
# Colour-blind check (Machado 2009 simulation): P2P blues vs filtered browns, contour set GT/T0/SARef/fixed and the
# PuOr diverging map stay distinguishable under protanopia and deuteranopia; the ground truth is drawn as a filled
# silhouette, not a line, so it is told apart from the pose contours by shape as well as colour.
plt.rcParams.update({
 "font.family":"Liberation Sans","font.size":7,"axes.titlesize":7.5,"axes.labelsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
 "legend.fontsize":7,"mathtext.fontset":"custom","mathtext.rm":"Liberation Sans","mathtext.it":"Liberation Sans:italic",
 "mathtext.bf":"Liberation Sans:bold","mathtext.sf":"Liberation Sans","pdf.fonttype":42,"ps.fonttype":42,
 "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"xtick.major.size":2.5,"ytick.major.size":2.5,
 "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#52606D","axes.labelcolor":C["ink"],
 "xtick.color":C["ink"],"ytick.color":C["ink"],"text.color":C["ink"],"lines.linewidth":1.1,"legend.frameon":False,
 "axes.titleweight":"bold","axes.titlelocation":"left","axes.titlepad":4,"savefig.dpi":600})
def panel_title(ax,s): ax.set_title(s,loc="left",fontsize=7.5,fontweight="bold",pad=4)
def save(fig,path):
    fig.savefig(path+".pdf",dpi=500); fig.savefig(path+".png",dpi=600)
