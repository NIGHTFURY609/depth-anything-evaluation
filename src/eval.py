"""Evaluation: alignment, metrics, subsets, confidence quality, filtering and bootstrap.

Pure numpy, so it runs without a GPU and is reused unchanged for the fine-tuned models.
"""

import cv2
import numpy as np
from scipy.stats import rankdata

from src.rellis import (
    GROUP_GRASS, GROUP_GROUND, GROUP_NAMES, GROUP_OTHER, GROUP_TREES, GROUP_VOID, terrain_groups,
)

ALIGN_METHODS = {"depth": ["median", "ls", "huber"], "disparity": ["ls", "huber"]}
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def valid_mask(depth, keep, dmin: float, dmax: float) -> np.ndarray:
    """Ground truth alone decides which pixels count, so every model is scored on the same set."""
    return keep & (depth >= dmin) & (depth <= dmax)


# --- alignment ---------------------------------------------------------------------------

def fit_median_scale(pred, gt) -> float:
    return float(np.median(gt / pred))


def fit_scale_shift_ls(x, y) -> tuple[float, float]:
    A = np.stack([x, np.ones_like(x)], axis=1).astype(np.float64)
    (s, t), *_ = np.linalg.lstsq(A, np.asarray(y, np.float64), rcond=None)
    return float(s), float(t)


def fit_scale_shift_huber(x, y, c: float = 1.345, iters: int = 50) -> tuple[float, float]:
    """IRLS from the least-squares fit. delta = c * sigma, sigma = 1.4826 * MAD of the residuals,
    re-estimated every iteration: textbook constants, nothing tuned on data."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    s, t = fit_scale_shift_ls(x, y)
    for _ in range(iters):
        r = y - (s * x + t)
        sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
        if sigma <= 1e-12 * max(np.abs(y).max(), 1.0):
            break
        w = np.minimum(1.0, c * sigma / np.maximum(np.abs(r), 1e-300))
        s_new, t_new = _weighted_ls(x, y, np.sqrt(w))
        converged = abs(s_new - s) <= 1e-10 * abs(s) and abs(t_new - t) <= 1e-10 * max(abs(t), 1e-12)
        s, t = s_new, t_new
        if converged:
            break
    return s, t


def _weighted_ls(x, y, sw) -> tuple[float, float]:
    A = np.stack([x * sw, sw], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, y * sw, rcond=None)
    return float(s), float(t)


def align(pred, kind: str, gt, mask, method: str, dmin: float, dmax: float):
    """Fits on the valid pixels and returns depth for the whole image, clamped to [dmin, dmax].

    `median` scales depth. `ls` and `huber` fit scale and shift in disparity space: on the
    model's disparity, or on the inverse of its depth. Non-positive disparity becomes dmax.
    """
    pred = pred.astype(np.float64)
    if method == "median":
        if kind != "depth":
            raise ValueError("median scaling needs a depth prediction")
        s, t = fit_median_scale(pred[mask], gt[mask]), 0.0
        raw = s * pred
    elif method in ("ls", "huber"):
        x = pred if kind == "disparity" else np.divide(1.0, pred, out=np.zeros_like(pred), where=pred > 0)
        fit = fit_scale_shift_ls if method == "ls" else fit_scale_shift_huber
        s, t = fit(x[mask], 1.0 / gt[mask])
        disp = s * x + t
        raw = np.divide(1.0, disp, out=np.full_like(disp, np.inf), where=disp > 0)
    else:
        raise ValueError(f"Unknown alignment: {method}")
    clamped = float(np.mean((raw[mask] < dmin) | (raw[mask] > dmax))) if mask.any() else np.nan
    return np.clip(raw, dmin, dmax), {"s": s, "t": t, "clamped": clamped}


# --- metrics and subsets -----------------------------------------------------------------

def pixel_metrics(pred, gt, phantom_ratio: float = 0.8, overshoot_ratio: float = 1.25) -> dict:
    n = int(np.size(gt))
    if n == 0:
        return {"n": 0, **{k: np.nan for k in ("absrel", "rmse", "delta1", "phantom", "overshoot")}}
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "n": n,
        "absrel": float(np.mean(np.abs(pred - gt) / gt)),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "phantom": float(np.mean(pred < phantom_ratio * gt)),
        "overshoot": float(np.mean(pred > overshoot_ratio * gt)),
    }


def subset_masks(valid, labels, edges: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Terrain groups, vegetation (grass and trees), and the edge band: its depth and semantic
    parts and their union, `edge`, which is the plan's definition."""
    groups = terrain_groups(labels)
    masks = {"all": valid}
    for group in (GROUP_GRASS, GROUP_TREES, GROUP_GROUND, GROUP_OTHER):
        masks[GROUP_NAMES[group]] = valid & (groups == group)
    masks["vegetation"] = masks["grass"] | masks["trees_bushes"]
    for name, edge in edges.items():
        masks[name] = valid & edge
    if edges:
        masks["edge"] = valid & np.logical_or.reduce(list(edges.values()))
    return masks


def edge_band(depth, points, labels, k: int, tau: float) -> dict[str, np.ndarray]:
    """Points with a neighbouring point (row +-1, column +-k) across a depth jump (`edge_depth`)
    or a group boundary (`edge_semantic`).

    Pass every LiDAR point in range, not only the occlusion-filtered ones: the filter removes
    exactly the far points beside a near edge. Restrict the result to the valid set afterwards.

    Neighbours never reach the next LiDAR ring (about 7 rows away), whose depth differs on flat
    ground anyway. So horizontal depth edges between rings are not detected. Grass and trees are
    one group here, so the grass/bush label disagreement does not create boundaries.
    """
    groups = terrain_groups(labels)
    merged = np.where(groups == GROUP_TREES, GROUP_GRASS, groups)
    height, width = depth.shape
    pad = lambda a, v: np.pad(a, ((1, 1), (k, k)), constant_values=v)
    d, ok, g = pad(depth.astype(np.float64), 0.0), pad(points, False), pad(merged, GROUP_VOID)
    jumps = np.zeros((height, width), bool)
    boundaries = np.zeros((height, width), bool)
    for dy in (-1, 0, 1):
        for dx in range(-k, k + 1):
            if dy == 0 and dx == 0:
                continue
            nd = d[1 + dy : 1 + dy + height, k + dx : k + dx + width]
            nok = ok[1 + dy : 1 + dy + height, k + dx : k + dx + width]
            ng = g[1 + dy : 1 + dy + height, k + dx : k + dx + width]
            both = points & nok
            hi, lo = np.maximum(depth, nd), np.minimum(depth, nd)
            jump = hi > (1 + tau) * lo
            boundary = (merged != ng) & (merged != GROUP_VOID) & (ng != GROUP_VOID)
            jumps |= both & jump
            boundaries |= both & boundary
    return {"edge_depth": jumps, "edge_semantic": boundaries}


# --- confidence quality ------------------------------------------------------------------

def _removal_curve(err_in_removal_order, fractions) -> np.ndarray:
    n = err_in_removal_order.size
    tail = np.concatenate([np.cumsum(err_in_removal_order[::-1])[::-1], [0.0]])
    removed = np.floor(np.asarray(fractions) * n).astype(int)
    return tail[removed] / (n - removed)


def sparsification(err, conf, fractions, rng) -> dict:
    """Mean error of the pixels left after removing the least confident fraction.

    Ties (DA3's confidence floor of 1.0) are broken randomly, not in raster order. The random
    curve is flat at the mean error in expectation, so AURG needs no sampling.
    """
    by_conf = np.lexsort((rng.random(err.size), conf))
    by_err = np.argsort(-err, kind="stable")
    conf_curve = _removal_curve(err[by_conf], fractions)
    oracle_curve = _removal_curve(err[by_err], fractions)
    random_curve = np.full_like(conf_curve, err.mean())
    return {
        "conf": conf_curve,
        "oracle": oracle_curve,
        "ause": float(_trapezoid(conf_curve - oracle_curve, fractions)),
        "aurg": float(_trapezoid(random_curve - conf_curve, fractions)),
    }


def auroc(score, positive, min_pos: int = 20, min_neg: int = 20) -> float:
    """Probability that a positive outscores a negative, ties counted half. NaN if either class
    is too small to rank."""
    positive = np.asarray(positive, bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos < min_pos or n_neg < min_neg:
        return np.nan
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --- filtering ---------------------------------------------------------------------------

def _sobel_magnitude(image) -> np.ndarray | None:
    """Offroad-Nav's gradient magnitude, 0-255: min-max to uint8, 5x5 Gaussian, 3x3 Sobel,
    scaled by its maximum. None for a constant image."""
    lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return None
    gray = ((image - lo) / (hi - lo) * 255.0).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mag = np.hypot(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
    top = mag.max()
    return mag / top * 255.0 if top > 0 else np.zeros_like(mag)


def _ellipse(thickness: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))


def sobel_mask(image, thickness: int) -> np.ndarray:
    """The Offroad-Nav edge mask (LARIAD/Offroad-Nav @ 69640cf, MIT): the magnitude above, Otsu
    threshold, elliptical dilation of diameter `thickness`. Pass disparity, or 1/depth."""
    mag = _sobel_magnitude(image)
    if mag is None:
        return np.zeros(image.shape, bool)
    _, edges = cv2.threshold(mag.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thickness > 1:
        edges = cv2.dilate(edges, _ellipse(thickness), iterations=1)
    return edges > 0


def sobel_score(image, thickness: int) -> np.ndarray:
    """The same magnitude, dilated like the mask, as a continuous score: each pixel takes the
    strongest gradient within the kernel. Removing the highest scores first is the Sobel filter
    at any chosen budget. Its Otsu threshold removes 73-86% of valid pixels on RELLIS val."""
    mag = _sobel_magnitude(image)
    if mag is None:
        return np.zeros(image.shape)
    return cv2.dilate(mag, _ellipse(thickness), iterations=1) if thickness > 1 else mag


def budget_remove(scores, n: int, candidates, rng) -> np.ndarray:
    """Removes the n lowest-scored candidate pixels, breaking ties randomly."""
    idx = np.flatnonzero(candidates)
    order = np.lexsort((rng.random(idx.size), scores.ravel()[idx]))
    removed = np.zeros(scores.size, bool)
    removed[idx[order[:n]]] = True
    return removed.reshape(scores.shape)


# --- bootstrap ---------------------------------------------------------------------------

def block_bootstrap(values, block: int, resamples: int, seed: int,
                    positions=None) -> tuple[float, float, float]:
    """Mean and 95% percentile interval, resampling blocks of consecutive images. NaNs are
    skipped, as in a nan-mean of each resample.

    `positions` are the images' indices in the split's full frame list. Blocks are cut on them,
    so a subset present only on some frames (image labels exist on every other one) still gets
    blocks of the same length in time.
    """
    v = np.asarray(values, np.float64)
    positions = np.arange(v.size) if positions is None else np.asarray(positions)
    _, ids = np.unique(positions // block, return_inverse=True)
    ok = ~np.isnan(v)
    sums = np.bincount(ids, np.where(ok, v, 0.0))
    counts = np.bincount(ids, ok.astype(np.float64))
    pick = np.random.default_rng(seed).integers(0, sums.size, (resamples, sums.size))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    mean = float(sums.sum() / counts.sum()) if counts.sum() else np.nan
    lo, hi = np.nanpercentile(means, [2.5, 97.5]) if np.isfinite(means).any() else (np.nan, np.nan)
    return mean, float(lo), float(hi)


def paired_block_bootstrap(a, b, block: int, resamples: int, seed: int,
                           positions=None) -> tuple[float, float, float]:
    """Mean difference a - b over images valid for both, with the same block resampling."""
    diff = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return block_bootstrap(diff, block, resamples, seed, positions)


# --- one frame, and the freeze gate ------------------------------------------------------

def evaluate_frame(pred, conf, kind: str, gt: dict, ecfg: dict, rng) -> dict:
    """Every per-image result for one prediction against one ground truth.

    Returns lists of rows: `accuracy` (per alignment, label source and subset), `confidence`
    and `curves` (depth models only), and `filtering`. `skipped` is set when too few pixels are
    valid to fit an alignment.
    """
    out = {"accuracy": [], "confidence": [], "curves": [], "filtering": [], "skipped": False}
    depth, labels = gt["depth"], gt["labels"]
    dmin, dmax = ecfg["min_depth"], ecfg["max_depth"]
    valid = valid_mask(depth, gt["keep"], dmin, dmax)
    if valid.sum() < ecfg["min_valid"]:
        out["skipped"] = True
        return out

    points = (depth >= dmin) & (depth <= dmax)
    edges = edge_band(depth, points, labels, ecfg["edge_k"], ecfg["edge_tau"])
    subsets = {"lidar": subset_masks(valid, labels, edges)}
    bins = ecfg["distance_bins"]
    for name in ("all", "grass", "trees_bushes", "ground", "vegetation"):
        for lo, hi in zip(bins[:-1], bins[1:]):
            subsets["lidar"][f"{name}|{lo}-{hi}m"] = subsets["lidar"][name] & (depth >= lo) & (depth < hi)
    if "image_labels" in gt:
        subsets["image"] = subset_masks(valid, gt["image_labels"], edges)

    filters = _filter_masks(pred, conf, kind, valid, ecfg, rng)
    fractions = np.arange(0, ecfg["sparsification_max"] + 1e-9, ecfg["sparsification_step"])
    ratios = {"phantom_ratio": ecfg["phantom_ratio"], "overshoot_ratio": ecfg["overshoot_ratio"]}

    for method in ALIGN_METHODS[kind]:
        aligned, info = align(pred, kind, depth, valid, method, dmin, dmax)
        for source, masks in subsets.items():
            for name, mask in masks.items():
                metrics = pixel_metrics(aligned[mask], depth[mask], **ratios)
                if metrics["n"] < ecfg["min_subset_px"]:
                    metrics = {**pixel_metrics(np.array([]), np.array([])), "n": metrics["n"]}
                out["accuracy"].append({"align": method, "label_source": source, "subset": name, **metrics, **info})

        err = np.abs(aligned[valid] - depth[valid]) / depth[valid]
        if conf is not None:
            c = conf[valid]
            curves = sparsification(err, c, fractions, rng)
            positive = err > ecfg["high_error"]
            out["confidence"].append({
                "align": method,
                "auroc": auroc(-c, positive, ecfg["auroc_min_class"], ecfg["auroc_min_class"]),
                "ause": curves["ause"],
                "aurg": curves["aurg"],
                "n_pos": int(positive.sum()),
                "n_neg": int((~positive).sum()),
            })
            out["curves"].append({"align": method, "conf": curves["conf"], "oracle": curves["oracle"]})

        rows = []
        for name, (group, removed) in filters.items():
            masks = removed if isinstance(removed, tuple) else (removed,)
            scores = [pixel_metrics(aligned[valid & ~m], depth[valid & ~m], **ratios) for m in masks]
            rows.append({
                "align": method,
                "filter": name,
                "group": group,
                "removed_valid": float(np.mean([(m & valid).sum() / valid.sum() for m in masks])),
                "removed_all": float(np.mean([m.mean() for m in masks])),
                "remaining": int(min(r["n"] for r in scores)),
                "absrel": float(np.mean([r["absrel"] for r in scores])),
                "phantom": float(np.mean([r["phantom"] for r in scores])),
            })
        # A filter that leaves too few pixels voids every filter it is compared with, so the
        # comparisons stay over the same images.
        for group in {r["group"] for r in rows}:
            members = [r for r in rows if r["group"] == group]
            if min(r["remaining"] for r in members) < ecfg["min_subset_px"]:
                for r in members:
                    r["absrel"] = r["phantom"] = np.nan
        out["filtering"] += rows
    return out


def _filter_masks(pred, conf, kind: str, valid, ecfg: dict, rng) -> dict:
    """Removal masks as {name: (comparison group, mask or tuple of masks)}, fixed before
    alignment so every alignment sees the same removals.

    Faithful Offroad-Nav mask, at each thickness: confidence removes as many valid pixels as
    Sobel did (`conf_valid`) or as many pixels of the whole image (`conf_all`, the deployment
    view, which may spend its budget on sky). Ranked filters, at each fixed share of valid
    pixels, the primary comparison: the highest Sobel scores (`sobelrank`), the lowest
    confidence (`confrank`), or at random (`randomrank`). Random removal is one mask per seed.
    """
    signal = pred if kind == "disparity" else np.divide(1.0, pred, out=np.zeros_like(pred), where=pred > 0)
    everything = np.ones_like(valid)

    def random(n):
        return tuple(budget_remove(rng.random(valid.shape), n, valid, rng)
                     for _ in range(ecfg["random_filter_seeds"]))

    masks = {"none": ("none", np.zeros_like(valid))}
    for th in ecfg["sobel_thickness"]:
        sobel = sobel_mask(signal, th)
        n_valid, n_all = int((sobel & valid).sum()), int(sobel.sum())
        group = f"sobel{th}"
        masks[f"sobel{th}"] = (group, sobel)
        if conf is not None:
            masks[f"conf_valid{th}"] = (group, budget_remove(conf, n_valid, valid, rng))
            masks[f"conf_all{th}"] = (group, budget_remove(conf, n_all, everything, rng))
        masks[f"random_valid{th}"] = (group, random(n_valid))
    score = sobel_score(signal, ecfg["sobel_rank_thickness"])
    for budget in ecfg["filter_budgets"]:
        n = int(round(budget * valid.sum()))
        group = f"rank{budget:g}"
        masks[f"sobelrank{budget:g}"] = (group, budget_remove(-score, n, valid, rng))
        if conf is not None:
            masks[f"confrank{budget:g}"] = (group, budget_remove(conf, n, valid, rng))
        masks[f"randomrank{budget:g}"] = (group, random(n))
    return masks


def check_can_evaluate(split: str, frozen: bool, dirty: bool) -> None:
    """The test split is evaluated only from a clean commit whose config says eval.frozen."""
    if split != "test":
        return
    if not frozen:
        raise SystemExit("refusing the test split: set eval.frozen: true in config.yaml first")
    if dirty:
        raise SystemExit("refusing the test split: commit or stash changes first")
