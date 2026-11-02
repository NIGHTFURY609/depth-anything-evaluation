"""LiDAR to camera projection, z-buffering and the occlusion filter."""

import cv2
import numpy as np

# Keep points slightly beyond the image edge before distorting. Further out the distortion
# polynomial stops being monotonic and can fold points back into the image.
_FOV_MARGIN = 1.2


def project_points(
    xyz: np.ndarray,
    T_cam_lidar: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    source_wh: tuple[int, int],
    target_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns float pixel coordinates at the target size, camera z, and indices of kept points.

    The source and target sizes may have slightly different aspect ratios, because DA3's
    input is resized non-uniformly to 504x322. Depth values are never resampled.
    """
    cam = xyz @ T_cam_lidar[:3, :3].T + T_cam_lidar[:3, 3]
    z = cam[:, 2]
    W, H = source_wh
    max_x = max(K[0, 2], W - K[0, 2]) / K[0, 0] * _FOV_MARGIN
    max_y = max(K[1, 2], H - K[1, 2]) / K[1, 1] * _FOV_MARGIN
    in_front = z > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        in_view = np.abs(cam[:, 0] / z) <= max_x
        in_view &= np.abs(cam[:, 1] / z) <= max_y
    kept = np.flatnonzero(in_front & in_view)

    if len(kept) == 0:
        return np.empty(0), np.empty(0), np.empty(0), kept
    pixels, _ = cv2.projectPoints(cam[kept].astype(np.float64), np.zeros(3), np.zeros(3), K, dist)
    pixels = pixels.reshape(-1, 2)
    # Pixel centres sit at integers, so scale about the top-left corner of the image.
    u = (pixels[:, 0] + 0.5) * target_wh[0] / W - 0.5
    v = (pixels[:, 1] + 0.5) * target_wh[1] / H - 0.5
    return u, v, z[kept], kept


def zbuffer(
    u: np.ndarray, v: np.ndarray, z: np.ndarray, labels: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest point per pixel. Depth 0 and label 0 mean empty."""
    cols, rows = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    cols, rows, z, labels = cols[inside], rows[inside], z[inside], labels[inside]

    far_to_near = np.argsort(-z, kind="stable")
    depth = np.zeros((height, width), dtype=np.float32)
    out_labels = np.zeros((height, width), dtype=np.uint8)
    depth[rows[far_to_near], cols[far_to_near]] = z[far_to_near]
    out_labels[rows[far_to_near], cols[far_to_near]] = labels[far_to_near]
    return depth, out_labels


def occlusion_mask(
    depth: np.ndarray, radius_x: int, radius_y: int, rel_threshold: float, abs_threshold: float
) -> np.ndarray:
    """True for valid pixels not hidden behind a much closer neighbour.

    The window covers the pixel's row and the rows above it. On ground the closer neighbours
    lie below, so including those rows would discard distant ground as if it were occluded.
    """
    filled = np.where(depth > 0, depth, np.inf).astype(np.float32)
    kernel = np.ones((radius_y + 1, 2 * radius_x + 1), dtype=np.uint8)
    nearest = cv2.erode(
        filled,
        kernel,
        anchor=(radius_x, radius_y),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=float("inf"),
    )
    allowed = np.maximum(rel_threshold * filled, abs_threshold)
    return (depth > 0) & (filled - nearest <= allowed)


def shift_agreement(lidar_labels: np.ndarray, image_labels: np.ndarray, radius: int) -> np.ndarray:
    """Agreement after moving the LiDAR labels by (dx, dy), indexed [dy + radius, dx + radius].

    A peak away from the centre means a systematic calibration offset of that many pixels.
    """
    size = 2 * radius + 1
    height, width = lidar_labels.shape
    grid = np.full((size, size), np.nan)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            moved = lidar_labels[
                max(0, -dy) : height - max(0, dy), max(0, -dx) : width - max(0, dx)
            ]
            target = image_labels[
                max(0, dy) : height - max(0, -dy), max(0, dx) : width - max(0, -dx)
            ]
            grid[dy + radius, dx + radius] = label_agreement(moved, target)
    return grid


def label_agreement(lidar_labels: np.ndarray, image_labels: np.ndarray) -> float:
    """Fraction of pixels labelled by both sources where the two agree; NaN without overlap."""
    both = (lidar_labels > 0) & (image_labels > 0)
    if not both.any():
        return float("nan")
    return float(np.mean(lidar_labels[both] == image_labels[both]))
