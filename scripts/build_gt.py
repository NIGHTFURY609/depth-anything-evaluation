"""Caches ground truth for every frame of every sequence to <cache_root>/gt/."""

import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.gt import build_frame, load_calibration, save_frame
from src.rellis import list_frames


def _process(frame, calibrations, cfg) -> None:
    save_frame(build_frame(frame, calibrations[frame.seq], cfg), frame, cfg["paths"]["cache_root"])


def main() -> None:
    cfg = load_config()
    sequences = cfg["gt"]["sequences"]
    calibrations = {seq: load_calibration(cfg, seq) for seq in sequences}
    frames = [f for seq in sequences for f in list_frames(cfg["paths"]["data_root"], seq)]
    worker = partial(_process, calibrations=calibrations, cfg=cfg)
    with Pool(8) as pool:
        for _ in tqdm(pool.imap_unordered(worker, frames, chunksize=16), total=len(frames)):
            pass
    print(f"cached {len(frames)} frames")


if __name__ == "__main__":
    main()
