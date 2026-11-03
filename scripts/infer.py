"""Caches zero-shot predictions for a split, and writes the evaluated frame list."""

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import git_hash, load_config, set_seed
from src.infer import eval_frames, load_runner, write_predictions

MODELS = ["da2_small", "da2_base", "da3_small", "da3_base"]


def write_frame_list(frames, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "frame"])
        writer.writerows((fr.seq, fr.frame) for fr in frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--stride", type=int, help="defaults to split.eval_stride")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    frames = eval_frames(cfg, args.split, args.stride)
    if args.split != "train":
        write_frame_list(frames, Path(cfg["paths"]["results_root"]) / "phase2" / f"frames_{args.split}.csv")
    commit = git_hash()
    for key in args.models:
        run, kind = load_runner(cfg, key, "cuda")
        paths = write_predictions(
            run, frames, cfg["paths"]["cache_root"], key, cfg["models"][key], kind, commit,
            cfg["input"]["process_res"],
        )
        print(f"{key}: {len(paths)} frames of {args.split} cached ({kind})", flush=True)
        del run
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
