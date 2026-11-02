"""Label-free alignment: score LiDAR depth edges against image edges over pixel shifts.

Works at full resolution, so offsets are in source pixels (divide by ~3.73 for 322-row pixels).
Horizontal-gradient and vertical-gradient scores are kept apart: vertical image gradients at
LiDAR points where depth changes between rings constrain dy, ring-wise jumps constrain dx.
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from multiprocessing import Pool
from src.config import load_config
from src.gt import load_calibration
from src.projection import project_points
from src.rellis import list_frames, load_scan

cv2.setNumThreads(0)
R = 30
STEP = 2
SHIFTS = np.arange(-R, R + 1, STEP)
cfg = load_config()
W, H = cfg["gt"]["source_size"]


def edge_points(xyz):
    """Ouster scans keep a structured order, 64 rings x 2048 columns, once empties are included."""
    return xyz


def score_frame(args):
    frame, calib = args
    raw = np.fromfile(frame.scan, dtype=np.float32).reshape(-1, 4)[:, :3]
    if len(raw) != 131072:
        return None
    grid = raw.reshape(64, 2048, 3)
    rng = np.linalg.norm(grid, axis=2)
    valid = rng > 0
    # Closer side of a depth jump, along the ring (constrains dx) and across rings (constrains dy).
    jump_col = np.zeros_like(rng, bool)
    a, b = rng[:, :-1], rng[:, 1:]
    both = valid[:, :-1] & valid[:, 1:]
    jump_col[:, :-1] |= both & (b - a > 0.5) & (b > 1.3 * a)
    jump_col[:, 1:] |= both & (a - b > 0.5) & (a > 1.3 * b)
    jump_row = np.zeros_like(rng, bool)
    a, b = rng[:-1, :], rng[1:, :]
    both = valid[:-1, :] & valid[1:, :]
    jump_row[:-1, :] |= both & (b - a > 0.5) & (b > 1.3 * a)
    jump_row[1:, :] |= both & (a - b > 0.5) & (a > 1.3 * b)

    image = cv2.imread(str(frame.image), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    image = cv2.GaussianBlur(image, (5, 5), 1.0)
    gx = np.abs(cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3))

    out = {}
    for name, mask, grad in (("col", jump_col, gx), ("row", jump_row, gy)):
        pts = grid[mask]
        if len(pts) < 20:
            out[name] = None
            continue
        u, v, _, _ = project_points(pts.astype(np.float64), calib.T_cam_lidar, calib.K, calib.dist, (W, H), (W, H))
        s = np.full((len(SHIFTS), len(SHIFTS)), np.nan)
        for i, dy in enumerate(SHIFTS):
            for j, dx in enumerate(SHIFTS):
                c = np.rint(u + dx).astype(int); r = np.rint(v + dy).astype(int)
                ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
                s[i, j] = grad[r[ok], c[ok]].mean() if ok.sum() > 10 else np.nan
        out[name] = s
    return out


def main():
    for seq in cfg["gt"]["sequences"]:
        calib = load_calibration(cfg, seq)
        frames = list_frames(cfg["paths"]["data_root"], seq)[::8]
        with Pool(8) as p:
            results = [r for r in p.imap_unordered(score_frame, [(f, calib) for f in frames], chunksize=4) if r]
        print(seq, "frames", len(results))
        for name in ("col", "row"):
            grids = [r[name] for r in results if r[name] is not None]
            g = np.nanmean(grids, axis=0)
            i, j = np.unravel_index(np.nanargmax(g), g.shape)
            zero = g[len(SHIFTS) // 2, len(SHIFTS) // 2]
            dy_profile = np.nanmean(g, axis=1); dx_profile = np.nanmean(g, axis=0)
            print(f"  {name}: n={len(grids)} best dx={SHIFTS[j]:+d} dy={SHIFTS[i]:+d} gain={g[i,j]/zero:.3f} | marginal best dx={SHIFTS[np.nanargmax(dx_profile)]:+d} dy={SHIFTS[np.nanargmax(dy_profile)]:+d}")
        # Half-splits, to see whether the estimate is stable.
        for name in ("col", "row"):
            grids = [r[name] for r in results if r[name] is not None]
            for part in (grids[::2], grids[1::2]):
                g = np.nanmean(part, axis=0)
                i, j = np.unravel_index(np.nanargmax(g), g.shape)
                print(f"    split {name}: dx={SHIFTS[j]:+d} dy={SHIFTS[i]:+d}")


if __name__ == "__main__":
    main()
