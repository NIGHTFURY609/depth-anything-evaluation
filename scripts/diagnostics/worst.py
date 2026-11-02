import sys
sys.path.insert(0, ".")
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.phase1_gate import draw_overlay
from src.config import load_config
from src.gt import frame_path
from src.rellis import list_frames

cfg = load_config()
d = pd.read_csv("results/phase1_agreement.csv", dtype={"seq": str})
print(d.agreement_all.describe(percentiles=[.001, .01, .05, .5]).round(3).to_string())
print("frames with agreement<0.5:", d[d.agreement_all < .5].groupby("seq").size().to_dict())
print("frames with points<3000:", d[d.points < 3000].groupby("seq").size().to_dict())
print(d[d.agreement_all < .5].groupby("seq").frame.agg(["min", "max"]).to_string())
pick = pd.concat([d.nsmallest(3, "agreement_all"), d.nsmallest(2, "points")]).drop_duplicates(["seq", "frame"])
print(pick.to_string())

# All frames, not just annotated: scan point count per pixel grid
low = []
for seq in cfg["gt"]["sequences"]:
    fs = list_frames(cfg["paths"]["data_root"], seq)
    n = []
    for f in fs:
        with np.load(frame_path(cfg["paths"]["cache_root"], seq, f.frame)) as g:
            n.append((f.frame, int((g["depth"] > 0).sum()), float(g["keep"].sum())))
    arr = np.array(n)
    print(seq, "points min/1%/median:", arr[:, 1].min(), np.percentile(arr[:, 1], 1).round(), np.median(arr[:, 1]),
          "kept fraction median:", np.median(arr[:, 2] / np.maximum(arr[:, 1], 1)).round(3))
    for row in arr[arr[:, 1] < 2000]:
        low.append((seq, int(row[0]), int(row[1])))
    m = {f.frame: f for f in fs}
    for _, r in pick[pick.seq == seq].iterrows():
        with np.load(frame_path(cfg["paths"]["cache_root"], seq, r.frame)) as g:
            draw_overlay(m[r.frame], dict(g), cfg, Path(f"results/overlays/worst_{seq}_{r.frame:06d}.jpg"))
print("frames with <2000 points:", len(low), low[:40])
