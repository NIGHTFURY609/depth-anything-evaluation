import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src.config import load_config
from src.gt import frame_path

cfg = load_config()
d = pd.read_csv("results/phase1_agreement.csv", dtype={"seq": str})
for seq, lo, hi in [("00002", 1235, 1285), ("00001", 460, 480)]:
    print(d[(d.seq == seq) & d.frame.between(lo, hi)][["frame", "agreement_all", "points"]].to_string(index=False))
for frame in (1265, 1255):
    with np.load(frame_path(cfg["paths"]["cache_root"], "00002", frame)) as g:
        L, I = g["labels"], g["image_labels"]
    both = (L > 0) & (I > 0)
    pairs, n = np.unique(np.stack([L[both], I[both]], 1), axis=0, return_counts=True)
    top = np.argsort(-n)[:6]
    print(frame, "(lidar,image) top pairs:", [(pairs[i].tolist(), int(n[i])) for i in top])
    print("  image classes:", dict(zip(*[x.tolist() for x in np.unique(I, return_counts=True)])))
# Long stretches of very low agreement anywhere?
low = d[d.agreement_all < 0.3].groupby("seq").frame.apply(list).to_dict()
print({k: v for k, v in low.items()})
# What confuses agreement overall, per sequence (in a sample)
for seq in cfg["gt"]["sequences"]:
    s = d[d.seq == seq].iloc[::20]
    tot = {}
    for fr in s.frame:
        with np.load(frame_path(cfg["paths"]["cache_root"], seq, fr)) as g:
            L, I = g["labels"], g["image_labels"]
        both = (L > 0) & (I > 0) & (L != I)
        p, n = np.unique(np.stack([L[both], I[both]], 1), axis=0, return_counts=True)
        for a, b in zip(p, n):
            tot[tuple(a.tolist())] = tot.get(tuple(a.tolist()), 0) + int(b)
    print(seq, "top disagreements (lidar,image):", sorted(tot.items(), key=lambda x: -x[1])[:5])
