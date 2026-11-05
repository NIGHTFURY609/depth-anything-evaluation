"""Runs a model over a split's frames and caches predictions to <cache_root>/pred/<model>/."""

from pathlib import Path

import numpy as np
import torch

from src.gt import frame_path
from src.models import forward_da2, forward_da3, load_da2, load_da3, load_finetuned, preprocess
from src.rellis import Frame, list_frames

# Batch 1 throughout: bf16 results can shift slightly with batch size, and Phase 4 must match.
BATCH = 1


def eval_frames(cfg: dict, split: str, stride: int | None = None) -> list[Frame]:
    """Every stride-th frame of each sequence in the split, minus logged drops."""
    seqs = cfg["split"][split]
    seqs = [seqs] if isinstance(seqs, str) else seqs
    stride = cfg["split"]["eval_stride"] if stride is None else stride
    dropped = cfg["gt"]["dropped_frames"]
    frames = []
    for seq in seqs:
        skip = set(dropped.get(seq, []))
        frames += [f for f in list_frames(cfg["paths"]["data_root"], seq)[::stride] if f.frame not in skip]
    return frames


def load_runner(cfg: dict, model_key: str, device: str, checkpoint=None):
    """Returns run(x) -> (pred, conf or None), and the kind of pred: depth or disparity.

    With a checkpoint, model_key is ignored and the fine-tuned DA3 is loaded instead.
    """
    if checkpoint is not None:
        net, _ = load_finetuned(cfg, checkpoint, device)

        def run(x):
            out = forward_da3(net, x)
            return out["depth"], out["conf"]

        return run, "depth"
    hf_id = cfg["models"][model_key]
    if model_key.startswith("da3"):
        net = load_da3(hf_id, device).model

        def run(x):
            out = forward_da3(net, x)
            return out["depth"], out["conf"]

        return run, "depth"

    model = load_da2(hf_id, device)
    return (lambda x: (forward_da2(model, x), None)), "disparity"


@torch.no_grad()
def write_predictions(
    run, frames: list[Frame], cache_root, model_key: str, model_id: str, kind: str,
    git_hash: str, process_res: int, device: str = "cuda",
) -> list[Path]:
    paths = []
    for frame in frames:
        x = preprocess([str(frame.image)], process_res).to(device)
        pred, conf = run(x)
        out = {
            "pred": pred[0].float().cpu().numpy(),
            "model_id": model_id,
            "kind": kind,
            "git_hash": git_hash,
        }
        if conf is not None:
            out["conf"] = conf[0].float().cpu().numpy()
        path = frame_path(cache_root, frame.seq, frame.frame, f"pred/{model_key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
        paths.append(path)
    return paths
