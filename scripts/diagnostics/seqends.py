"""Are any two sequences continuous? Compares the last frame of each with the first of every other.

Mean absolute grey-level difference at 240x150; a consecutive-frame baseline is printed for scale.
"""
import sys
sys.path.insert(0, ".")
import cv2, numpy as np
from src.config import load_config
from src.rellis import list_frames
cfg = load_config()
seqs = cfg["gt"]["sequences"]
fr = {s: list_frames(cfg["paths"]["data_root"], s) for s in seqs}
def img(f): return cv2.resize(cv2.imread(str(f.image), cv2.IMREAD_GRAYSCALE), (240, 150)).astype(np.float32)
ends = {}
for s in seqs:
    ends[s] = {"first": [img(f) for f in fr[s][:3]], "last": [img(f) for f in fr[s][-3:]]}
    print(s, "first", fr[s][0].image.name, "last", fr[s][-1].image.name)
# mean abs diff between last frame of a and first frame of b; also a within-sequence consecutive baseline
for s in seqs:
    print(s, "consecutive-frame baseline MAD", round(float(np.mean(np.abs(ends[s]["first"][0]-ends[s]["first"][1]))),1))
for a in seqs:
    for b in seqs:
        if a == b: continue
        d = np.mean(np.abs(ends[a]["last"][-1] - ends[b]["first"][0]))
        flag = "  <-- near duplicate" if d < 12 else ""
        print(f"last {a} -> first {b}: MAD {d:5.1f}{flag}")
