"""First look at grass errors, on val only: panels of the frames where DA3-S does worst on grass.

Each panel: the image, the aligned prediction, and LiDAR points on grass coloured by signed
relative error (red: predicted too close, the phantom direction; blue: too far). Written to
results/overlays/ (git-ignored, derived from RELLIS images). Test frames are never drawn.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.eval import align, valid_mask
from src.gt import frame_path
from src.rellis import GROUP_GRASS, list_frames, terrain_groups

MODEL, ALIGN, N = "da3_small", "median", 6


def main() -> None:
    cfg = load_config()
    ecfg = cfg["eval"]
    seq = cfg["split"]["val"]
    acc = pd.read_csv("results/phase2/val/accuracy_per_image.csv.gz", dtype={"seq": str})
    rows = acc[(acc["model"] == MODEL) & (acc["align"] == ALIGN) & (acc["gt_source"] == "gt")
               & (acc["label_source"] == "lidar") & (acc["subset"] == "grass")]
    worst = rows.nlargest(N, "absrel")
    print(worst[["frame", "absrel", "phantom", "overshoot", "n"]].to_string(index=False))
    frames = {f.frame: f for f in list_frames(cfg["paths"]["data_root"], seq)}
    w, h = cfg["input"]["width"], cfg["input"]["height"]
    for frame in worst["frame"]:
        cache = cfg["paths"]["cache_root"]
        with np.load(frame_path(cache, seq, frame, f"pred/{MODEL}")) as p:
            pred = p["pred"]
        with np.load(frame_path(cache, seq, frame)) as g:
            depth, labels, keep = g["depth"], g["labels"], g["keep"]
        valid = valid_mask(depth, keep, ecfg["min_depth"], ecfg["max_depth"])
        aligned, _ = align(pred, "depth", depth, valid, ALIGN, ecfg["min_depth"], ecfg["max_depth"])
        image = cv2.resize(cv2.imread(str(frames[frame].image)), (w, h), interpolation=cv2.INTER_AREA)
        log = np.log(aligned / ecfg["min_depth"]) / np.log(ecfg["max_depth"] / ecfg["min_depth"])
        depth_vis = cv2.applyColorMap((np.clip(log, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        errors = (image * 0.5).astype(np.uint8)
        grass = valid & (terrain_groups(labels) == GROUP_GRASS)
        for r, c in np.argwhere(grass):
            e = np.clip((aligned[r, c] - depth[r, c]) / depth[r, c] / 0.5, -1, 1)  # +-50% saturates
            colour = (0, 0, int(255 * -e)) if e < 0 else (int(255 * e), 0, 0)
            cv2.circle(errors, (int(c), int(r)), 1, tuple(int(v) for v in np.maximum(colour, 40)), -1)
        panel = np.hstack([image, depth_vis, errors])
        cv2.imwrite(f"results/overlays/grass_val_{seq}_{frame:06d}.jpg", panel)


if __name__ == "__main__":
    main()
