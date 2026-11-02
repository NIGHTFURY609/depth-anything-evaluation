"""Small rotation corrections to the camera-LiDAR calibration, fitted from labels and geometry.

No depth model is involved, so the correction cannot favour any model being compared. The fit
scores how many LiDAR points of sharply bounded classes land on the same class in the image
annotation, over a grid of pixel shifts.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# A shift counts as near-optimal within this fraction of the peak. Points inside a large object
# score the same over a range of shifts, so the centre of that plateau is the better estimate.
_PLATEAU = 0.98


def frame_counts(
    labels: np.ndarray, keep: np.ndarray, image_labels: np.ndarray, classes: list[int], radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """Grids of (same-class hits, annotated points) indexed [dy + radius, dx + radius]."""
    size = 2 * radius + 1
    hits, totals = np.zeros((size, size)), np.zeros((size, size))
    rows, cols = np.nonzero(keep & np.isin(labels, classes))
    if len(rows) == 0:
        return hits, totals
    point_labels = labels[rows, cols]
    height, width = labels.shape
    for dy in range(-radius, radius + 1):
        moved_rows = rows + dy
        for dx in range(-radius, radius + 1):
            moved_cols = cols + dx
            inside = (
                (moved_rows >= 0) & (moved_rows < height) & (moved_cols >= 0) & (moved_cols < width)
            )
            found = image_labels[moved_rows[inside], moved_cols[inside]]
            annotated = found > 0
            hits[dy + radius, dx + radius] = np.count_nonzero(
                found[annotated] == point_labels[inside][annotated]
            )
            totals[dy + radius, dx + radius] = np.count_nonzero(annotated)
    return hits, totals


def _precision(hits: np.ndarray, totals: np.ndarray) -> np.ndarray:
    return np.divide(hits, totals, out=np.zeros_like(hits, dtype=float), where=totals > 0)


def estimate_offset(hits: np.ndarray, totals: np.ndarray, radius: int) -> tuple[float, float]:
    """The (dx, dy) to apply to the LiDAR points: the centre of the near-optimal plateau."""
    precision = _precision(hits, totals)
    rows, cols = np.nonzero(precision >= precision.max() * _PLATEAU)
    return float(cols.mean() - radius), float(rows.mean() - radius)


def precision_at(hits: np.ndarray, totals: np.ndarray, radius: int, dx: float, dy: float) -> float:
    row, col = int(round(dy)) + radius, int(round(dx)) + radius
    return float(_precision(hits, totals)[row, col])


def block_split(count: int, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Alternating blocks of frames, because neighbouring frames are nearly identical."""
    indices = np.arange(count)
    in_first = (indices // block) % 2 == 0
    return indices[in_first], indices[~in_first]


def bootstrap_offsets(
    hits: np.ndarray, totals: np.ndarray, radius: int, block: int, resamples: int, seed: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    """95% interval of (dx, dy), resampling blocks of frames. hits and totals are per frame."""
    count = len(hits)
    starts = np.arange(0, count, block)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        chosen = rng.choice(starts, size=len(starts), replace=True)
        indices = np.concatenate([np.arange(s, min(s + block, count)) for s in chosen])
        estimates.append(estimate_offset(hits[indices].sum(0), totals[indices].sum(0), radius))
    low = np.percentile(estimates, 2.5, axis=0)
    high = np.percentile(estimates, 97.5, axis=0)
    return (float(low[0]), float(low[1])), (float(high[0]), float(high[1]))


def decide(
    dx: float,
    dy: float,
    ci_low: tuple[float, float],
    ci_high: tuple[float, float],
    holdout_zero: float,
    holdout_shifted: float,
    points: int,
    max_ci_width: float,
    min_points: int,
) -> dict:
    """Accepts a correction only if it is stable and improves frames it was not fitted on."""
    if points < min_points:
        reason = "too_few_points"
    elif max(ci_high[0] - ci_low[0], ci_high[1] - ci_low[1]) > max_ci_width:
        reason = "unstable"
    elif holdout_shifted <= holdout_zero:
        reason = "no_holdout_gain"
    else:
        reason = "ok"
    return {"dx": dx, "dy": dy, "accepted": reason == "ok", "reason": reason}


def rotation_for_pixel_shift(
    dx: float, dy: float, K: np.ndarray, source_wh: tuple[int, int], target_wh: tuple[int, int]
) -> np.ndarray:
    """A camera-frame rotation moving projected points by (dx, dy) target pixels at any depth.

    A rotation, because a pixel shift that holds at every range cannot come from a translation.
    """
    pixels_per_radian_x = K[0, 0] * target_wh[0] / source_wh[0]
    pixels_per_radian_y = K[1, 1] * target_wh[1] / source_wh[1]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(
        "xy", [-dy / pixels_per_radian_y, dx / pixels_per_radian_x]
    ).as_matrix()
    return T
