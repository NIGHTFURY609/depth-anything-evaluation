"""Caches ground truth for every frame of every sequence to <cache_root>/gt/."""

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.gt import build_frame, load_calibration, save_frame
from src.rellis import list_frames


def _process(frame, calibrations, cfg, subdir) -> None:
    gt = build_frame(frame, calibrations[frame.seq], cfg)
    save_frame(gt, frame, cfg["paths"]["cache_root"], subdir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="apply the estimated corrections and write to gt_sensitivity instead of gt",
    )
    args = parser.parse_args()

    cv2.setNumThreads(0)  # One thread per worker; the pool already uses the cores.
    cfg = load_config()
    sequences = cfg["gt"]["sequences"]
    corrections = cfg["gt"]["sensitivity_calibration"] if args.sensitivity else None
    subdir = "gt_sensitivity" if args.sensitivity else "gt"
    calibrations = {seq: load_calibration(cfg, seq, corrections) for seq in sequences}
    frames = [f for seq in sequences for f in list_frames(cfg["paths"]["data_root"], seq)]
    worker = partial(_process, calibrations=calibrations, cfg=cfg, subdir=subdir)
    with Pool(8) as pool:
        for _ in tqdm(pool.imap_unordered(worker, frames, chunksize=16), total=len(frames)):
            pass
    print(f"cached {len(frames)} frames to {subdir}")


if __name__ == "__main__":
    main()
