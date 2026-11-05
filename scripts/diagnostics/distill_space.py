"""Phase 3: should the distillation L1 use linear or log depth?

Rule fixed before running (DECISIONS.md, 2026-11-06): on the cached zero-shot DA3-S val predictions,
take r = d_t / median(d_t) per image, and the share of sum(r) held by the image's farthest 10% of
pixels. A linear L1 on r weights each pixel's relative change by r, so a heavy far tail means the
loss is mostly about far and sky pixels. If the median share is above 0.3, distil in log space.
Uniform depth gives 0.1. Uses val only and no student, so it cannot favour any trained model.
"""
import sys

sys.path.insert(0, ".")
import numpy as np

from src.config import load_config
from src.gt import frame_path
from src.infer import eval_frames

THRESHOLD = 0.3
cfg = load_config()
shares, tops, maxes = [], [], []
for f in eval_frames(cfg, "val"):
    with np.load(frame_path(cfg["paths"]["cache_root"], f.seq, f.frame, "pred/da3_small")) as p:
        d = p["pred"].astype(np.float64).ravel()
    r = np.sort(d / np.median(d))
    top = r[int(0.9 * r.size):]
    shares.append(top.sum() / r.sum())
    tops.append(top.min())
    maxes.append(r[-1])
shares = np.array(shares)
print(f"val frames: {len(shares)}")
print("share of sum(r) in farthest 10% of pixels: "
      f"median {np.median(shares):.3f}, p10 {np.quantile(shares, 0.1):.3f}, p90 {np.quantile(shares, 0.9):.3f}")
print(f"r at the 90th percentile: median {np.median(tops):.2f}; max r: median {np.median(maxes):.1f}, "
      f"p90 {np.quantile(maxes, 0.9):.1f}")
print(f"rule: log space if median share > {THRESHOLD}")
print("decision:", "log" if np.median(shares) > THRESHOLD else "linear")
