"""Fits a per-sequence pixel correction from sharply bounded classes.

Scores the official-calibration cache. Writes results/phase1_refine.csv with each sequence's
stability verdict, and all point estimates to gt.sensitivity_calibration.
"""

import csv
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calib_refine import (
    block_split,
    bootstrap_offsets,
    decide,
    estimate_offset,
    frame_counts,
    precision_at,
)
from src.config import load_config, set_seed
from src.gt import frame_path
from src.rellis import list_frames


def _counts(frame, cache_root, classes, radius):
    with np.load(frame_path(cache_root, frame.seq, frame.frame)) as gt:
        return frame_counts(gt["labels"], gt["keep"], gt["image_labels"], classes, radius)


def refine_sequence(seq: str, cfg: dict, settings: dict) -> dict:
    radius = settings["radius"]
    frames = [
        f for f in list_frames(cfg["paths"]["data_root"], seq) if f.image_labels is not None
    ]
    worker = partial(
        _counts,
        cache_root=cfg["paths"]["cache_root"],
        classes=settings["classes"],
        radius=radius,
    )
    with Pool(8) as pool:
        results = list(tqdm(pool.imap(worker, frames, chunksize=8), total=len(frames), desc=seq))
    hits = np.stack([r[0] for r in results])
    totals = np.stack([r[1] for r in results])
    # Only frames that contain these classes inform the fit.
    used = totals[:, radius, radius] > 0
    hits, totals = hits[used], totals[used]
    points = int(totals[:, radius, radius].sum())
    row = {"seq": seq, "frames_used": int(used.sum()), "points": points}
    if points == 0:
        return {**row, **decide(0.0, 0.0, (0, 0), (0, 0), 0, 0, 0, 0, settings["min_points"])}

    block = settings["block"]
    fit, holdout = block_split(len(hits), block)
    dx_fit, dy_fit = estimate_offset(hits[fit].sum(0), totals[fit].sum(0), radius)
    holdout_hits, holdout_totals = hits[holdout].sum(0), totals[holdout].sum(0)
    holdout_zero = precision_at(holdout_hits, holdout_totals, radius, 0, 0)
    holdout_shifted = precision_at(holdout_hits, holdout_totals, radius, dx_fit, dy_fit)

    dx, dy = estimate_offset(hits.sum(0), totals.sum(0), radius)
    low, high = bootstrap_offsets(
        hits, totals, radius, block, settings["resamples"], cfg["seed"]
    )
    decision = decide(
        dx, dy, low, high, holdout_zero, holdout_shifted, points,
        settings["max_ci_width"], settings["min_points"],
    )
    return {
        **row,
        "fit_dx": round(dx_fit, 2),
        "fit_dy": round(dy_fit, 2),
        "ci_dx": f"{low[0]:.1f}..{high[0]:.1f}",
        "ci_dy": f"{low[1]:.1f}..{high[1]:.1f}",
        "holdout_zero": round(holdout_zero, 4),
        "holdout_shifted": round(holdout_shifted, 4),
        **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in decision.items()},
    }


def main() -> None:
    cv2.setNumThreads(0)
    cfg = load_config()
    set_seed(cfg["seed"])
    settings = cfg["gt"]["refinement"]
    rows = [refine_sequence(seq, cfg, settings) for seq in cfg["gt"]["sequences"]]

    results = Path(cfg["paths"]["results_root"])
    fields = sorted({k for row in rows for k in row}, key=lambda k: list(rows[-1]).index(k))
    with open(results / "phase1_refine.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # Every point estimate is kept, accepted or not: they drive the sensitivity analysis only.
    estimates = {r["seq"]: {"dx": float(r["dx"]), "dy": float(r["dy"])} for r in rows}
    with open(cfg["gt"]["sensitivity_calibration"], "w") as f:
        yaml.safe_dump(estimates, f)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
