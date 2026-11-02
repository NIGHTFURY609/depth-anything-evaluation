"""Per-class shift analysis: precision of LiDAR class points on the same image class vs (dx, dy)."""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.config import load_config
from src.gt import frame_path
from src.rellis import list_frames

R = 14
CLASSES = {17: "person", 8: "vehicle", 27: "barrier", 5: "pole", 18: "fence", 12: "building", 9: "object", 15: "log", 34: "rubble", 4: "tree", 19: "bush", 3: "grass", 23: "concrete", 7: "sky"}
cfg = load_config()
H, W = cfg["input"]["height"], cfg["input"]["width"]
hit = {c: np.zeros((2 * R + 1, 2 * R + 1)) for c in CLASSES}
tot = {c: np.zeros((2 * R + 1, 2 * R + 1)) for c in CLASSES}
per_seq = {}
for seq in cfg["gt"]["sequences"]:
    fs = [f for f in list_frames(cfg["paths"]["data_root"], seq) if f.image_labels is not None]
    sh = {c: np.zeros((2 * R + 1, 2 * R + 1)) for c in CLASSES}
    st = {c: np.zeros((2 * R + 1, 2 * R + 1)) for c in CLASSES}
    for f in fs[::2]:
        with np.load(frame_path(cfg["paths"]["cache_root"], seq, f.frame)) as g:
            L, I, K = g["labels"], g["image_labels"], g["keep"]
        rows, cols = np.nonzero(K & (L > 0))
        labs = L[rows, cols]
        for c in np.intersect1d(np.unique(labs), list(CLASSES)):
            m = labs == c
            r0, c0 = rows[m], cols[m]
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    rr, cc = r0 + dy, c0 + dx
                    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                    vals = I[rr[ok], cc[ok]]
                    sh[c][dy + R, dx + R] += (vals == c).sum()
                    st[c][dy + R, dx + R] += (vals > 0).sum()
    per_seq[seq] = (sh, st)
    for c in CLASSES:
        hit[c] += sh[c]; tot[c] += st[c]

def line(name, h, t):
    if t[R, R] < 200:
        return f"  {name:9s} n<200"
    g = h / np.maximum(t, 1)
    r, col = np.unravel_index(np.argmax(g), g.shape)
    return f"  {name:9s} n={int(t[R,R]):8d} zero={g[R,R]:.3f} best={g[r,col]:.3f} dx={col-R:+d} dy={r-R:+d}  dy-profile@dx0: " + " ".join(f"{g[R+d,R]:.2f}" for d in (-6,-3,0,3,6,9,12))
print("overall precision (LiDAR class points landing on same image class)")
for c, n in CLASSES.items():
    print(line(n, hit[c], tot[c]))
print("per sequence, compact classes only")
for seq, (sh, st) in per_seq.items():
    print(seq)
    for c in (17, 8, 27, 5, 18, 12):
        print(line(CLASSES[c], sh[c], st[c]))
