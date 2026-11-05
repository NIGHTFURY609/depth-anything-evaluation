"""Caches predictions for a split, and writes the evaluated frame list.

Zero-shot models by config key, or a fine-tuned checkpoint under its own name:
    python scripts/infer.py --split val --checkpoint ~/da3ft-data/checkpoints/<run>/best_val.pt --name <run>
"""

import argparse
import csv
import gc
import hashlib
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
    parser.add_argument("--checkpoint", help="a Phase 3 checkpoint; predictions go to pred/<name>/")
    parser.add_argument("--name", help="cache name for --checkpoint; must not be a config model key")
    args = parser.parse_args()
    if args.checkpoint and (not args.name or args.name in MODELS):
        parser.error("--checkpoint needs a --name that is not a zero-shot model key")

    cfg = load_config()
    set_seed(cfg["seed"])
    commit = git_hash()  # before writing the frame list, which would make the tree dirty
    frames = eval_frames(cfg, args.split, args.stride)
    if args.split != "train":
        write_frame_list(frames, Path(cfg["paths"]["results_root"]) / "phase2" / f"frames_{args.split}.csv")
    if args.checkpoint:
        sha = hashlib.sha256(Path(args.checkpoint).expanduser().read_bytes()).hexdigest()[:12]
        base = cfg["models"][cfg["train"]["base_model"]]
        jobs = [(args.name, f"{base}+{args.name}@{sha}", Path(args.checkpoint).expanduser())]
    else:
        jobs = [(key, cfg["models"][key], None) for key in args.models]
    for key, model_id, checkpoint in jobs:
        run, kind = load_runner(cfg, key, "cuda", checkpoint)
        paths = write_predictions(
            run, frames, cfg["paths"]["cache_root"], key, model_id, kind, commit,
            cfg["input"]["process_res"],
        )
        print(f"{key}: {len(paths)} frames of {args.split} cached ({kind})", flush=True)
        del run
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
