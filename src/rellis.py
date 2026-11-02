"""RELLIS-3D loader: image<->scan pairing, calibration, labels and terrain groups."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

GROUP_VOID, GROUP_GRASS, GROUP_TREES, GROUP_GROUND, GROUP_OTHER = range(5)
GROUP_NAMES = {
    GROUP_VOID: "void",
    GROUP_GRASS: "grass",
    GROUP_TREES: "trees_bushes",
    GROUP_GROUND: "ground",
    GROUP_OTHER: "other",
}
# Ontology IDs. The plan's "gravel" is not a RELLIS class.
_GROUP_CLASSES = {
    GROUP_VOID: [0],
    GROUP_GRASS: [3],
    GROUP_TREES: [4, 19],
    GROUP_GROUND: [1, 10, 23, 33],
}
_FRAME_RE = re.compile(r"frame(\d+)-")
_EXTRINSICS_KEY = "os1_cloud_node-pylon_camera_node"


@dataclass(frozen=True)
class Frame:
    seq: str
    frame: int
    image: Path
    scan: Path
    point_labels: Path
    image_labels: Path | None


def list_frames(data_root, seq: str) -> list[Frame]:
    """Pairs each image with the scan of the same frame number, as RELLIS's own example does."""
    base = Path(data_root) / "Rellis-3D" / seq
    frames = []
    for image in (base / "pylon_camera_node").glob("frame*.jpg"):
        number = int(_FRAME_RE.match(image.name).group(1))
        scan = base / "os1_cloud_node_kitti_bin" / f"{number:06d}.bin"
        point_labels = base / "os1_cloud_node_semantickitti_label_id" / f"{number:06d}.label"
        if not (scan.exists() and point_labels.exists()):
            continue
        image_labels = base / "pylon_camera_node_label_id" / f"{image.stem}.png"
        frames.append(
            Frame(
                seq, number, image, scan, point_labels,
                image_labels if image_labels.exists() else None,
            )
        )
    return sorted(frames, key=lambda f: f.frame)


def load_scan(scan_path, label_path) -> tuple[np.ndarray, np.ndarray]:
    """Returns xyz (N, 3) and class IDs (N). Ouster returns all zeros for a missing echo."""
    points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    labels = np.fromfile(label_path, dtype=np.uint32)
    if len(labels) != len(points):
        raise ValueError(f"{len(points)} points but {len(labels)} labels in {label_path}")
    has_return = np.any(points != 0, axis=1)
    return points[has_return], (labels[has_return] & 0xFFFF).astype(np.uint8)


def load_intrinsics(path) -> np.ndarray:
    fx, fy, cx, cy = np.loadtxt(path)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def load_extrinsics(path) -> np.ndarray:
    """Returns the LiDAR->camera transform. The file stores camera->LiDAR."""
    with open(path) as f:
        data = yaml.safe_load(f)[_EXTRINSICS_KEY]
    q, t = data["q"], data["t"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T[:3, 3] = [t["x"], t["y"], t["z"]]
    return np.linalg.inv(T)


def terrain_groups(labels: np.ndarray) -> np.ndarray:
    groups = np.full(labels.shape, GROUP_OTHER, dtype=np.uint8)
    for group, classes in _GROUP_CLASSES.items():
        groups[np.isin(labels, classes)] = group
    return groups
