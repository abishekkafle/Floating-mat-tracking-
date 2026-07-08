"""Generate publication figures using the Wong (2011) Nature-standard colorblind-safe palette."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

WONG = {
    "black":  "#000000", "orange": "#E69F00", "sky":    "#56B4E9",
    "green":  "#009E73", "yellow": "#F0E442", "blue":   "#0072B2",
    "verm":   "#D55E00", "purple": "#CC79A7",
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222", "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "figure.dpi": 110, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "axes.spines.top": False, "axes.spines.right": False,
})

FILES = {
    "Tension-guide":        "/mnt/project/ftw_B_TensionGuide_TPosts_withwaves__Copy_2.csv",
    "Distributed-modular":  "/mnt/project/ftw_C_DistributedModular_withwaves__Copy_2.csv",
}
COLORS = {"Tension-guide": WONG["verm"], "Distributed-modular": WONG["blue"]}

dfs = {}
for label, path in FILES.items():
    d = pd.read_csv(path, parse_dates=["timestamp"])
    d["elapsed_min"] = (d["timestamp"] - d["timestamp"].iloc[0]).dt.total_seconds() / 60.0
    d["r_smooth"]    = d["r_cm"].rolling(window=30, center=True, min_periods=5).mean()
    d["head_smooth"] = d["heading_deg"].rolling(window=30, center=True, min_periods=5).mean()
    dfs[label] = d

# Figure 4: time-series
fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
ax = axes[0]
for label, d in dfs.items():
    c = COLORS[label]
    ax.plot(d["elapsed_min"], d["r_cm"], color=c, lw=0.5, alpha=0.28)
    ax.plot(d["elapsed_min"], d["r_smooth"], color=c, lw=1.6, label=label)
ax.set_ylabel("Radial drift  $r$  (cm)")
ax.set_xlim(0, 30); ax.set_ylim(0, None)
ax.grid(True, ls="-", lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)
ax.text(0.012, 0.94, "a", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
ax.legend(loc="upper right", frameon=False, handlelength=1.6, ncol=2)

ax = axes[1]
for label, d in dfs.items():
    c = COLORS[label]
    ax.plot(d["elapsed_min"], d["heading_deg"], color=c, lw=0.5, alpha=0.28)
    ax.plot(d["elapsed_min"], d["head_smooth"], color=c, lw=1.6, label=label)
ax.axhline(0, color="#777777", lw=0.5, alpha=0.7)
ax.set_xlabel("Elapsed time (min)"); ax.set_ylabel("Heading deviation (deg)")
ax.set_xlim(0, 30)
ax.grid(True, ls="-", lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)
ax.text(0.012, 0.94, "b", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
fig.tight_layout(h_pad=0.6)
fig.savefig("/home/claude/Fig4_timeseries.pdf")
fig.savefig("/home/claude/Fig4_timeseries.png")
plt.close(fig)
print("Wrote Fig4_timeseries.{pdf,png}")

# Figure 5: KPI bars
with open("/home/claude/results.json") as f:
    res = json.load(f)
labels = ["Tension-guide", "Distributed-modular"]
keys = ["B_TensionGuide", "C_DistributedModular"]

def get(metric, lo_hi_key):
    vals = [res["per_config"][k][metric] for k in keys]
    lo = [res["per_config"][k][lo_hi_key][0] for k in keys]
    hi = [res["per_config"][k][lo_hi_key][1] for k in keys]
    return vals, lo, hi

mean_drift, md_lo, md_hi = get("mean_drift_cm", "mean_drift_ci")
p95,        p95_lo, p95_hi = get("p95_drift_cm",  "p95_drift_ci")
rms_h,      rh_lo,  rh_hi  = get("rms_heading_deg","rms_heading_ci")

def err_pair(vals, los, his):
    return [[v - lo for v, lo in zip(vals, los)],
            [hi - v for v, hi in zip(vals, his)]]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
x = np.arange(len(labels)); bar_colors = [COLORS[l] for l in labels]
panels = [
    (axes[0], mean_drift, md_lo, md_hi, "Mean radial drift (cm)", "a"),
    (axes[1], p95,        p95_lo, p95_hi, "P95 radial drift (cm)", "b"),
    (axes[2], rms_h,      rh_lo,  rh_hi,  "RMS heading (deg)",     "c"),
]
for ax, vals, los, his, ylabel, tag in panels:
    ax.bar(x, vals, color=bar_colors, edgecolor="#222222", lw=0.6, width=0.62,
           yerr=err_pair(vals, los, his), capsize=4,
           error_kw=dict(lw=0.9, ecolor="#222222"))
    ax.set_xticks(x)
    ax.set_xticklabels(["Tension-\nguide", "Distributed-\nmodular"], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.text(0.02, 0.97, tag, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    ax.grid(True, axis="y", ls="-", lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)
    ax.set_ylim(0, max(his) * 1.18)
fig.tight_layout()
fig.savefig("/home/claude/Fig5_kpi.pdf")
fig.savefig("/home/claude/Fig5_kpi.png")
plt.close(fig)
print("Wrote Fig5_kpi.{pdf,png}")

# Supplementary trajectory — shared axes for direct visual comparison + hexbin density
from matplotlib.colors import LinearSegmentedColormap

# Shared limit so the two panels are directly comparable
lim = max(
    max(abs(d["x_cm"]).max(), abs(d["y_cm"]).max()) for d in dfs.values()
) * 1.10

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharex=True, sharey=True)
hexes = []
for ax, (label, d) in zip(axes, dfs.items()):
    c = COLORS[label]
    # Per-config colormap from white to the config color
    cmap = LinearSegmentedColormap.from_list(f"w2{label}", ["#FFFFFF", c])
    hb = ax.hexbin(d["x_cm"], d["y_cm"], gridsize=22, cmap=cmap,
                   mincnt=1, linewidths=0.2, edgecolors="#FFFFFF")
    ax.set_aspect("equal")
    ax.axhline(0, color="#666666", lw=0.4); ax.axvline(0, color="#666666", lw=0.4)
    ax.set_xlabel("x (cm)")
    ax.set_title(label, fontsize=9, color=c, fontweight="bold")
    ax.grid(True, ls="-", lw=0.3, color="#EEEEEE"); ax.set_axisbelow(True)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    hexes.append(hb)
axes[0].set_ylabel("y (cm)")
# Single colorbar for sample count
cb = fig.colorbar(hexes[1], ax=axes, shrink=0.82, pad=0.025, aspect=22)
cb.set_label("Samples per cell")
cb.outline.set_linewidth(0.5)
fig.savefig("/home/claude/FigS_trajectory.pdf")
fig.savefig("/home/claude/FigS_trajectory.png")
plt.close(fig)
print("Wrote FigS_trajectory.{pdf,png}")
