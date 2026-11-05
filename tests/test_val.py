import math

import numpy as np
import pytest

from src.eval import evaluate_frame
from src.val import SUBSETS, frame_masks, frame_metrics, mean_metrics

ECFG = {
    "min_depth": 1.0, "max_depth": 80.0, "min_valid": 200, "min_subset_px": 50, "edge_k": 4,
    "edge_tau": 0.2, "distance_bins": [0, 10, 20, 80], "high_error": 0.25, "auroc_min_class": 20,
    "phantom_ratio": 0.8, "overshoot_ratio": 1.25, "sparsification_step": 0.05,
    "sparsification_max": 0.95, "sobel_thickness": [5], "random_filter_seeds": 2,
    "filter_budgets": [0.2], "sobel_rank_thickness": 5,
}


def scene(seed: int, h=60, w=90):
    """Sparse rings over grass (near and far), trees and ground, with some depth jumps."""
    rng = np.random.default_rng(seed)
    depth = np.zeros((h, w), np.float32)
    rows = np.arange(2, h, 3)
    for r in rows:
        depth[r] = 3 + (h - r) * 0.6 + rng.uniform(0, 1, w)
    depth[rows[:4], 30:50] = 6.0  # a near object in front of the far rows: depth jumps
    labels = np.where(depth > 0, 3, 0).astype(np.uint8)  # grass
    labels[:, 60:] = np.where(depth[:, 60:] > 0, 4, 0)  # trees
    labels[-6:, :] = np.where(depth[-6:, :] > 0, 1, 0)  # dirt
    keep = (depth > 0) & (rng.random((h, w)) > 0.05)
    return {"depth": depth, "labels": labels, "keep": keep}


def prediction(gt, seed: int):
    rng = np.random.default_rng(seed)
    pred = np.where(gt["depth"] > 0, gt["depth"], 30.0) * rng.uniform(0.7, 1.3, gt["depth"].shape) * 0.04
    conf = rng.uniform(1, 5, gt["depth"].shape)
    return pred.astype(np.float32), conf.astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_evaluate_frame(seed):
    gt = scene(seed)
    pred, conf = prediction(gt, seed + 10)
    ours = frame_metrics(pred, conf, gt["depth"], frame_masks(gt, ECFG), ECFG)
    rows = evaluate_frame(pred, conf, "depth", gt, ECFG, np.random.default_rng(0))["accuracy"]
    ref = {r["subset"]: r["absrel"] for r in rows if r["align"] == "median" and r["label_source"] == "lidar"}
    checked = 0
    for key, subset in SUBSETS.items():
        if math.isnan(ref[subset]):
            assert math.isnan(ours[key]), key
        else:
            assert ours[key] == pytest.approx(ref[subset], rel=1e-9), key
            checked += 1
    assert checked >= 4  # the scene must exercise most subsets


def test_skipped_frame_has_no_masks():
    gt = scene(0)
    gt["keep"][:] = False
    assert frame_masks(gt, ECFG) is None


def test_mean_skips_nan_and_counts_images():
    rows = [{"absrel_all": 0.1, "absrel_grass": float("nan")}, {"absrel_all": 0.3, "absrel_grass": 0.2}]
    out = mean_metrics(rows)
    assert out["absrel_all"] == pytest.approx(0.2)
    assert out["absrel_grass"] == pytest.approx(0.2)
    assert out["n_absrel_grass"] == 1


def test_confidence_summary():
    gt = scene(0)
    pred, _ = prediction(gt, 3)
    conf = np.full_like(pred, 1.01)
    out = frame_metrics(pred, conf, gt["depth"], frame_masks(gt, ECFG), ECFG)
    assert out["conf_floor"] == 1.0
    assert out["conf_median"] == pytest.approx(1.01)
