"""Estimate the LiDAR pixel offset by matching against zero-shot DA3 depth.

For sequence 00002 (the proposed test sequence) this prints only the location of the optimum,
never an error value, so no test-set accuracy is seen.
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
import torch
from src.config import load_config
from src.gt import frame_path
from src.models import forward_da3, load_da3, preprocess
from src.rellis import GROUP_GRASS, GROUP_GROUND, GROUP_TREES, list_frames, terrain_groups

TEST_SEQ = "00002"
R = 12
STRIDE = 15
cfg = load_config()
H, W = cfg["input"]["height"], cfg["input"]["width"]
net = load_da3(cfg["models"]["da3_small"], "cuda").model


def add(grid, count, pred, rows, cols, depth):
    """Scale-aligned mean absolute log error between shifted LiDAR and prediction."""
    if len(rows) < 50:
        return
    log_p = np.log(pred)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            r, c = rows + dy, cols + dx
            ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            diff = np.log(depth[ok]) - log_p[r[ok], c[ok]]
            grid[dy + R, dx + R] += np.mean(np.abs(diff - np.median(diff)))
            count[dy + R, dx + R] += 1


for seq in cfg["gt"]["sequences"]:
    frames = list_frames(cfg["paths"]["data_root"], seq)[::STRIDE]
    subsets = ("all", "non_ground", "edges", "first_half", "second_half")
    grids = {k: np.zeros((2 * R + 1, 2 * R + 1)) for k in subsets}
    counts = {k: np.zeros((2 * R + 1, 2 * R + 1)) for k in subsets}
    for i, f in enumerate(frames):
        x = preprocess([str(f.image)]).to("cuda")
        with torch.no_grad():
            pred = forward_da3(net, x)["depth"][0].float().cpu().numpy()
        with np.load(frame_path(cfg["paths"]["cache_root"], seq, f.frame)) as g:
            depth, labels, keep = g["depth"], g["labels"], g["keep"]
        rows, cols = np.nonzero(keep)
        d = depth[rows, cols]
        groups = terrain_groups(labels[rows, cols])
        add(grids["all"], counts["all"], pred, rows, cols, d)
        half = "first_half" if i % 2 == 0 else "second_half"
        add(grids[half], counts[half], pred, rows, cols, d)
        m = ~np.isin(groups, [GROUP_GRASS, GROUP_GROUND])
        add(grids["non_ground"], counts["non_ground"], pred, rows[m], cols[m], d[m])
        # Points at a LiDAR depth edge: a much farther valid pixel nearby.
        filled = np.where(depth > 0, depth, 0).astype(np.float32)
        far = cv2.dilate(filled, np.ones((9, 9), np.uint8))
        e = (far[rows, cols] > d * 1.3)
        add(grids["edges"], counts["edges"], pred, rows[e], cols[e], d[e])
    print(seq, "frames", len(frames))
    for k in subsets:
        g = grids[k] / np.maximum(counts[k], 1)
        g[counts[k] == 0] = np.nan
        r, c = np.unravel_index(np.nanargmin(g), g.shape)
        at_edge = r in (0, 2 * R) or c in (0, 2 * R)
        line = f"  {k:12s} best dx={c-R:+d} dy={r-R:+d}{' (AT SEARCH EDGE)' if at_edge else ''}"
        if seq != TEST_SEQ:
            line += f"  improvement over zero shift: {1 - g[r, c] / g[R, R]:.1%}"
        print(line)
