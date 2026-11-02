"""Builds per-frame ground truth at the model's resolution from a LiDAR scan."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.projection import occlusion_mask, project_points, zbuffer
from src.rellis import Frame, load_extrinsics, load_intrinsics, load_scan


@dataclass(frozen=True)
class Calibration:
    K: np.ndarray
    T_cam_lidar: np.ndarray
    dist: np.ndarray


def load_calibration(cfg: dict, seq: str) -> Calibration:
    """Extrinsics differ between sequences, so always load them per sequence."""
    calib = Path(cfg["paths"]["data_root"]) / "calib"
    return Calibration(
        K=load_intrinsics(calib / "intrinsics/Rellis-3D" / seq / "camera_info.txt"),
        T_cam_lidar=load_extrinsics(calib / "basler_to_ouster/Rellis_3D" / seq / "transforms.yaml"),
        dist=np.array(cfg["gt"]["distortion"], dtype=np.float64),
    )


def build_frame(frame: Frame, calib: Calibration, cfg: dict) -> dict[str, np.ndarray]:
    """Returns depth, labels and keep (occlusion-filtered mask), plus image_labels if annotated."""
    width, height = cfg["input"]["width"], cfg["input"]["height"]
    xyz, labels = load_scan(frame.scan, frame.point_labels)
    u, v, z, kept = project_points(
        xyz, calib.T_cam_lidar, calib.K, calib.dist, tuple(cfg["gt"]["source_size"]), (width, height)
    )
    depth, pixel_labels = zbuffer(u, v, z, labels[kept], height, width)
    gt = {
        "depth": depth,
        "labels": pixel_labels,
        "keep": occlusion_mask(depth, **cfg["gt"]["occlusion"]),
    }
    if frame.image_labels is not None:
        annotation = cv2.imread(str(frame.image_labels), cv2.IMREAD_UNCHANGED)
        gt["image_labels"] = cv2.resize(
            annotation, (width, height), interpolation=cv2.INTER_NEAREST_EXACT
        )
    return gt


def frame_path(cache_root, seq: str, frame: int) -> Path:
    return Path(cache_root) / "gt" / seq / f"{frame:06d}.npz"


def save_frame(gt: dict[str, np.ndarray], frame: Frame, cache_root) -> Path:
    path = frame_path(cache_root, frame.seq, frame.frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **gt)
    return path
