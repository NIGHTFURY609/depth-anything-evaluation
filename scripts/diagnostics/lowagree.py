"""Why do the remaining low-agreement frames disagree? Labels only, no depth model.

Prints the top LiDAR->image label pairs where they disagree, and writes a panel per frame:
the depth overlay on the left, the image annotation on the right with LiDAR points green
where the labels agree and red where they do not.
"""
import sys
sys.path.insert(0, ".")
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts.phase1_gate import draw_overlay
from src.config import load_config
from src.gt import frame_path
from src.rellis import list_frames

FRAMES = {"00001": [894, 896, 898], "00002": [100, 102, 104, 106, 108, 1325, 1327, 1331, 1335, 1345]}
cfg = load_config()
out = Path("results/overlays")
for seq, numbers in FRAMES.items():
    frames = {f.frame: f for f in list_frames(cfg["paths"]["data_root"], seq)}
    for n in numbers:
        f = frames[n]
        with np.load(frame_path(cfg["paths"]["cache_root"], seq, n)) as g:
            gt = dict(g)
        lidar, image = gt["labels"], gt["image_labels"]
        both = (lidar > 0) & (image > 0)
        bad = both & (lidar != image)
        pairs = Counter(zip(lidar[bad].tolist(), image[bad].tolist())).most_common(3)
        print(f"{seq} {n:6d} agree={1 - bad.sum() / max(both.sum(), 1):.3f} top pairs (lidar, image): {pairs}")
        draw_overlay(f, gt, cfg, out / "tmp_overlay.jpg")
        left = cv2.imread(str(out / "tmp_overlay.jpg"))
        right = cv2.applyColorMap((image.astype(np.uint8) * 7), cv2.COLORMAP_JET)
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
        for row, col in np.argwhere(both):
            colour = (0, 0, 255) if bad[row, col] else (0, 255, 0)
            cv2.circle(right, (int(col * 2), int(row * 2)), 1, colour, -1)
        cv2.imwrite(str(out / f"lowagree_{seq}_{n:06d}.jpg"), np.hstack([left, right]))
(out / "tmp_overlay.jpg").unlink()
