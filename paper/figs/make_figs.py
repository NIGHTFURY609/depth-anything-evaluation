"""Figures for the zero-shot evaluation paper. All numbers are copied from
paper-results.md (sections 4.3, 4.6, 4.7, 5.1, 5.2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# Fixed categorical order: DA3-S, DA3-B, DA2-S, DA2-B
C = {"DA3-S": "#2a78d6", "DA3-B": "#eb6834", "DA2-S": "#1baf7a", "DA2-B": "#eda100"}
GRAY = "#6b6b66"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#d9d9d4", "grid.linewidth": 0.4, "axes.grid": True, "axes.grid.axis": "y",
    "axes.axisbelow": True, "pdf.fonttype": 42,
})
COL_W = 3.45  # IEEE column width (in)

# ---------------- F3: sparsification curves (median alignment) ----------------
frac = np.arange(0, 1.0, 0.05)
s_conf = [0.1545,0.1306,0.1161,0.1069,0.0999,0.0943,0.0896,0.0858,0.0825,0.0798,0.0777,0.0759,0.0744,0.0731,0.0722,0.0712,0.0702,0.0690,0.0678,0.0669]
s_orac = [0.1545,0.1168,0.1007,0.0894,0.0802,0.0724,0.0655,0.0593,0.0536,0.0484,0.0434,0.0386,0.0341,0.0297,0.0254,0.0212,0.0170,0.0128,0.0086,0.0043]
b_conf = [0.1424,0.1174,0.1014,0.0924,0.0863,0.0814,0.0776,0.0745,0.0720,0.0702,0.0692,0.0686,0.0683,0.0682,0.0683,0.0683,0.0685,0.0690,0.0699,0.0716]
b_orac = [0.1424,0.1021,0.0862,0.0759,0.0679,0.0612,0.0554,0.0503,0.0456,0.0412,0.0371,0.0331,0.0293,0.0256,0.0219,0.0182,0.0146,0.0110,0.0073,0.0037]
# ls alignment, DA3-S, for the inset comparison
s_conf_ls = [0.1348,0.1116,0.1007,0.0933,0.0870,0.0815,0.0767,0.0725,0.0688,0.0656,0.0627,0.0599,0.0574,0.0550,0.0528,0.0507,0.0485,0.0464,0.0443,0.0429]

fig, ax = plt.subplots(figsize=(COL_W, 2.75))
ax.axhline(0.1545, color=C["DA3-S"], lw=0.7, ls=":", alpha=0.7)
ax.axhline(0.1424, color=C["DA3-B"], lw=0.7, ls=":", alpha=0.7)
ax.plot(frac*100, s_conf, color=C["DA3-S"], label="DA3-S, confidence")
ax.plot(frac*100, s_orac, color=C["DA3-S"], ls="--", label="DA3-S, oracle")
ax.plot(frac*100, b_conf, color=C["DA3-B"], label="DA3-B, confidence")
ax.plot(frac*100, b_orac, color=C["DA3-B"], ls="--", label="DA3-B, oracle")
ax.plot(frac*100, s_conf_ls, color=C["DA3-S"], lw=0.8, ls="-.", label="DA3-S, confidence (LS align.)")
ax.set_xlabel("Valid pixels removed (%)")
ax.set_ylabel("AbsRel on remaining pixels")
ax.set_xlim(0, 95); ax.set_ylim(0, 0.165)
ax.text(93, 0.1545+0.003, "random (DA3-S)", ha="right", va="bottom", fontsize=6, color=GRAY)
ax.text(93, 0.1424-0.003, "random (DA3-B)", ha="right", va="top", fontsize=6, color=GRAY)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=2.4, columnspacing=1.2)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_sparsification.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---------------- F4: AbsRel vs % removed, filters ----------------
budgets = [0, 10, 20, 30, 50]
conf_s = [0.155, 0.116, 0.100, 0.090, 0.078]
sob_s  = [0.155, 0.148, 0.151, 0.157, 0.173]
rnd_s  = [0.155, 0.155, 0.155, 0.155, 0.155]
conf_b = [0.142, 0.101, 0.086, 0.078, 0.069]
sob_b  = [0.142, 0.136, 0.139, 0.144, 0.163]
rnd_b  = [0.142, 0.142, 0.142, 0.142, 0.142]

fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.45), sharey=True)
for ax, name, conf, sob, rnd, otsu in [
    (axes[0], "DA3-S", conf_s, sob_s, rnd_s, (77.9, 0.445, 0.085, 0.161)),
    (axes[1], "DA3-B", conf_b, sob_b, rnd_b, (79.2, 0.391, 0.079, 0.147)),
]:
    col = C[name]
    ax.plot(budgets, sob, color="#b3261e", marker="s", ms=3.5, label="Sobel-ranked")
    ax.plot(budgets, conf, color=col, marker="o", ms=3.5, label="Confidence-ranked")
    ax.plot(budgets, rnd, color=GRAY, ls=":", marker="^", ms=3.5, label="Random")
    x, so, co, ro = otsu
    ax.scatter([x], [so], marker="s", s=22, facecolor="none", edgecolor="#b3261e", lw=1.0, zorder=5)
    ax.scatter([x], [co], marker="o", s=22, facecolor="none", edgecolor=col, lw=1.0, zorder=5)
    ax.scatter([x], [ro], marker="^", s=22, facecolor="none", edgecolor=GRAY, lw=1.0, zorder=5)
    ax.annotate("faithful Otsu\nmask (own\nbudget)", xy=(x, so), xytext=(x-5, so-0.10),
                ha="right", va="top", fontsize=6, color=GRAY,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=GRAY))
    ax.set_title(name, loc="left", fontweight="bold")
    ax.set_xlabel("Valid pixels removed (%)")
    ax.set_xlim(-3, 88); ax.set_ylim(0, 0.47)
    ax.set_xticks([0, 10, 20, 30, 50, 78])
axes[0].set_ylabel("AbsRel on remaining pixels")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=3, handlelength=2.2, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0.09, 1, 1))
fig.savefig(os.path.join(OUT, "fig_filtering.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---------------- F5: phantom & overshoot by distance (vegetation) ----------------
bins = ["0–10", "10–20", "20–80"]
models = ["DA3-S", "DA3-B", "DA2-S", "DA2-B"]
phantom = {"DA3-S": [0.009, 0.283, 0.702], "DA3-B": [0.009, 0.167, 0.489],
           "DA2-S": [0.015, 0.308, 0.665], "DA2-B": [0.017, 0.306, 0.621]}
over = {"DA3-S": [0.083, 0.072, 0.027], "DA3-B": [0.072, 0.097, 0.053],
        "DA2-S": [0.058, 0.097, 0.042], "DA2-B": [0.052, 0.098, 0.060]}
# 95% CIs available only for 10-20 (DA3-S) and 20-80 phantom
ci = {"DA3-S": [None, (0.223, 0.354), (0.617, 0.784)], "DA3-B": [None, None, (0.397, 0.594)],
      "DA2-S": [None, None, (0.563, 0.756)], "DA2-B": [None, None, (0.511, 0.724)]}

fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.1), gridspec_kw={"width_ratios": [1, 1]})
x = np.arange(len(bins)); w = 0.2
for ax, data, title in [(axes[0], phantom, "Phantom: pred $<0.8\\,g$"),
                        (axes[1], over, "Overshoot: pred $>1.25\\,g$")]:
    for i, m in enumerate(models):
        xs = x + (i - 1.5) * w
        ax.bar(xs, data[m], width=w*0.92, color=C[m], label=m, edgecolor="white", lw=0.5)
        if ax is axes[0]:
            for j, c in enumerate(ci[m]):
                if c:
                    ax.errorbar(xs[j], data[m][j], yerr=[[data[m][j]-c[0]], [c[1]-data[m][j]]],
                                fmt="none", ecolor="#333", elinewidth=0.6, capsize=1.5)
    ax.set_xticks(x); ax.set_xticklabels(bins); ax.set_xlabel("LiDAR depth (m)")
    ax.set_title(title, loc="left", fontsize=7.5)
axes[0].set_ylabel("Share of vegetation pixels")
axes[0].set_ylim(0, 0.85); axes[1].set_ylim(0, 0.12)
axes[0].legend(frameon=False, loc="upper left", ncol=1)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(os.path.join(OUT, "fig_distance.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("done")
