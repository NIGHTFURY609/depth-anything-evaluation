"""Val ground-truth statistics that fix the Phase 2 eval thresholds. Uses no depth model.

Depth distribution (for the range cap), spacing of points along a ring (for the edge band's k),
the share of each group flagged as edge for candidate (k, tau), and subset sizes per image.
"""
import sys
sys.path.insert(0, ".")
import numpy as np

from src.config import load_config
from src.eval import edge_band, subset_masks, valid_mask
from src.gt import frame_path
from src.infer import eval_frames

cfg = load_config()
frames = eval_frames(cfg, "val")
depths, gaps = [], []
counts = {name: [] for name in ("all", "grass", "trees_bushes", "ground", "other", "vegetation")}
CANDIDATES = [(k, tau) for k in (4, 6, 8) for tau in (0.15, 0.2, 0.3)]
flagged = {c: {g: [0, 0] for g in ("grass", "trees_bushes", "ground", "other", "all")} for c in CANDIDATES}
for f in frames[::2]:
    with np.load(frame_path(cfg["paths"]["cache_root"], f.seq, f.frame)) as g:
        depth, labels, keep = g["depth"], g["labels"], g["keep"]
    depths.append(depth[keep])
    for row in depth:
        cols = np.flatnonzero(row > 0)
        gaps.append(np.diff(cols))
    valid = valid_mask(depth, keep, 1.0, 80.0)
    points = (depth >= 1.0) & (depth <= 80.0)
    subsets = subset_masks(valid, labels, {})
    for name in counts:
        counts[name].append(int(subsets[name].sum()))
    for k, tau in CANDIDATES:
        e = edge_band(depth, points, labels, k, tau)
        edge = (e["edge_depth"] | e["edge_semantic"]) & valid
        for name in flagged[(k, tau)]:
            flagged[(k, tau)][name][0] += int((edge & subsets[name]).sum())
            flagged[(k, tau)][name][1] += int(subsets[name].sum())

d = np.concatenate(depths)
print(f"val frames used: {len(frames[::2])} (every 2nd evaluated frame)")
print("kept depth percentiles 0.1/1/50/99/99.9:", np.percentile(d, [0.1, 1, 50, 99, 99.9]).round(2))
print(f"fraction below 1 m: {np.mean(d < 1):.5f}, above 80 m: {np.mean(d > 80):.5f}, max {d.max():.1f}")
for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 80)]:
    print(f"  {lo:2d}-{hi:2d} m: {np.mean((d >= lo) & (d < hi)):.3f}")
gap = np.concatenate(gaps)
print("along-row gap between points, percentiles 25/50/75/90:", np.percentile(gap, [25, 50, 75, 90]))
print("edge share of valid pixels by group, (k, tau):")
for c in CANDIDATES:
    print(f"  k={c[0]} tau={c[1]:.2f}: " + "  ".join(f"{g} {a / max(b, 1):.3f}" for g, (a, b) in flagged[c].items()))
print("pixels per image, percentiles 5/25/50 and share of images with >= 100:")
for name, v in counts.items():
    v = np.array(v)
    print(f"  {name:13s} {np.percentile(v, [5, 25, 50]).round()}  {np.mean(v >= 100):.2f}")
