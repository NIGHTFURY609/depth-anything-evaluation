"""Fast validation for training: the frozen eval's median-aligned AbsRel on a few subsets.

Built from src/eval.py's own functions, so it gives the same per-image numbers as scripts/eval.py
for these subsets (tests/test_val.py). Selection uses `absrel_all` only; the rest are logged.
"""

import math

import numpy as np
import torch

from src.eval import align, edge_band, pixel_metrics, subset_masks, valid_mask
from src.gt import frame_path
from src.models import forward_da3
from src.train_data import load_resized, normalize, to_unit

SUBSETS = {
    "absrel_all": "all",
    "absrel_grass": "grass",
    "absrel_vegetation": "vegetation",
    "absrel_edge_depth": "edge_depth",
    "absrel_grass_far": "grass|20-80m",
}
CONF_FLOOR = 1.05  # DA3's confidence is exp(x) + 1, so values near 1 sit at its floor


def frame_masks(gt: dict, ecfg: dict) -> dict | None:
    """Valid pixels and the logged subsets; None when eval would skip the image."""
    depth, labels = gt["depth"], gt["labels"]
    dmin, dmax = ecfg["min_depth"], ecfg["max_depth"]
    valid = valid_mask(depth, gt["keep"], dmin, dmax)
    if valid.sum() < ecfg["min_valid"]:
        return None
    points = (depth >= dmin) & (depth <= dmax)
    masks = subset_masks(valid, labels, edge_band(depth, points, labels, ecfg["edge_k"], ecfg["edge_tau"]))
    for name in list(masks):
        if name in ("all", "grass", "trees_bushes", "ground", "vegetation"):
            bins = ecfg["distance_bins"]
            for lo, hi in zip(bins[:-1], bins[1:]):
                masks[f"{name}|{lo}-{hi}m"] = masks[name] & (depth >= lo) & (depth < hi)
    return {"valid": valid, **{key: masks[sub] for key, sub in SUBSETS.items()}}


def frame_metrics(pred, conf, depth, masks: dict, ecfg: dict) -> dict:
    dmin, dmax = ecfg["min_depth"], ecfg["max_depth"]
    aligned, _ = align(pred, "depth", depth, masks["valid"], "median", dmin, dmax)
    out = {}
    for key in SUBSETS:
        m = masks[key]
        n = int(m.sum())
        out[key] = pixel_metrics(aligned[m], depth[m])["absrel"] if n >= ecfg["min_subset_px"] else math.nan
    c = conf[masks["valid"]]
    out["conf_floor"] = float(np.mean(c < CONF_FLOOR))
    out["conf_median"] = float(np.median(c))
    return out


def mean_metrics(rows: list[dict]) -> dict:
    """Mean over images, NaN images skipped, as src/report.py summarises."""
    out = {}
    for key in rows[0]:
        values = np.array([r[key] for r in rows], dtype=np.float64)
        ok = ~np.isnan(values)
        out[key] = float(values[ok].mean()) if ok.any() else math.nan
        if key in SUBSETS:
            out[f"n_{key}"] = int(ok.sum())
    return out


class Validator:
    """Holds the val images (resized uint8) and masks in memory; scores a net at batch 1, as
    src/infer.py does."""

    def __init__(self, cfg: dict, frames: list, gt_source: str = "gt"):
        if any(f.seq != cfg["split"]["val"] for f in frames):
            raise ValueError("validation frames must come from the val sequence")
        self.ecfg = cfg["eval"]
        self.items = []
        for f in frames:
            with np.load(frame_path(cfg["paths"]["cache_root"], f.seq, f.frame, gt_source)) as g:
                gt = dict(g)
            masks = frame_masks(gt, self.ecfg)
            if masks is None:
                continue
            image = load_resized(str(f.image), cfg["input"]["process_res"])
            self.items.append((image, gt["depth"], masks))

    @torch.no_grad()
    def __call__(self, net, device: str = "cuda") -> dict:
        was_training = net.training
        net.eval()
        rows = []
        for image, depth, masks in self.items:
            x = normalize(to_unit(image))[None].to(device)
            out = forward_da3(net, x)
            pred = out["depth"][0].float().cpu().numpy()
            conf = out["conf"][0].float().cpu().numpy()
            rows.append(frame_metrics(pred, conf, depth, masks, self.ecfg))
        net.train(was_training)
        return mean_metrics(rows)
