import numpy as np
import pytest

from src.rellis import (
    GROUP_GROUND,
    GROUP_GRASS,
    GROUP_OTHER,
    GROUP_TREES,
    GROUP_VOID,
    list_frames,
    load_extrinsics,
    load_intrinsics,
    load_scan,
    terrain_groups,
)


def make_frame(root, seq, frame, annotated=False, scan=True):
    base = root / "Rellis-3D" / seq
    for folder in (
        "pylon_camera_node",
        "pylon_camera_node_label_id",
        "os1_cloud_node_kitti_bin",
        "os1_cloud_node_semantickitti_label_id",
    ):
        (base / folder).mkdir(parents=True, exist_ok=True)
    stem = f"frame{frame:06d}-1581624663_149"
    (base / "pylon_camera_node" / f"{stem}.jpg").touch()
    if annotated:
        (base / "pylon_camera_node_label_id" / f"{stem}.png").touch()
    if scan:
        (base / "os1_cloud_node_kitti_bin" / f"{frame:06d}.bin").touch()
        (base / "os1_cloud_node_semantickitti_label_id" / f"{frame:06d}.label").touch()


def test_list_frames_pairs_by_frame_number_in_order(tmp_path):
    make_frame(tmp_path, "00000", 12)
    make_frame(tmp_path, "00000", 3, annotated=True)
    frames = list_frames(tmp_path, "00000")
    assert [f.frame for f in frames] == [3, 12]
    assert frames[0].scan.name == "000003.bin"
    assert frames[0].point_labels.name == "000003.label"
    assert frames[0].image.name.startswith("frame000003-")
    assert frames[0].image_labels.name.endswith(".png")
    assert frames[1].image_labels is None


def test_list_frames_skips_image_without_scan(tmp_path):
    make_frame(tmp_path, "00000", 1)
    make_frame(tmp_path, "00000", 2, scan=False)
    assert [f.frame for f in list_frames(tmp_path, "00000")] == [1]


def test_load_scan_drops_empty_returns_and_keeps_labels_aligned(tmp_path):
    points = np.array(
        [[1, 2, 3, 0.5], [0, 0, 0, 0], [4, 5, 6, 0.1]], dtype=np.float32
    )
    bin_path, label_path = tmp_path / "s.bin", tmp_path / "s.label"
    points.tofile(bin_path)
    # The upper 16 bits hold the instance id; the class is the lower 16.
    np.array([(7 << 16) | 3, 0, 19], dtype=np.uint32).tofile(label_path)
    xyz, labels = load_scan(bin_path, label_path)
    assert xyz.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert labels.tolist() == [3, 19]


def test_load_scan_rejects_mismatched_label_count(tmp_path):
    bin_path, label_path = tmp_path / "s.bin", tmp_path / "s.label"
    np.zeros((2, 4), dtype=np.float32).tofile(bin_path)
    np.zeros(3, dtype=np.uint32).tofile(label_path)
    with pytest.raises(ValueError, match="labels"):
        load_scan(bin_path, label_path)


def test_load_intrinsics(tmp_path):
    path = tmp_path / "camera_info.txt"
    path.write_text("2813.6 2808.3 969.2 624.0\n")
    K = load_intrinsics(path)
    assert K.tolist() == [[2813.6, 0, 969.2], [0, 2808.3, 624.0], [0, 0, 1]]


def test_load_extrinsics_inverts_the_file_transform(tmp_path):
    # The file gives camera->LiDAR. A 90 degree turn about z plus a translation:
    path = tmp_path / "transforms.yaml"
    s = np.sqrt(0.5)
    path.write_text(
        "os1_cloud_node-pylon_camera_node:\n"
        f"  q: {{w: {s}, x: 0, y: 0, z: {s}}}\n"
        "  t: {x: 1, y: 2, z: 3}\n"
    )
    T = load_extrinsics(path)
    # The camera origin sits at t in the LiDAR frame, so T must send t to the origin.
    assert T @ np.array([1, 2, 3, 1]) == pytest.approx(np.array([0, 0, 0, 1]))


def test_terrain_groups():
    labels = np.array([0, 3, 4, 19, 1, 33, 10, 23, 7, 18, 34])
    assert terrain_groups(labels).tolist() == [
        GROUP_VOID,
        GROUP_GRASS,
        GROUP_TREES,
        GROUP_TREES,
        GROUP_GROUND,
        GROUP_GROUND,
        GROUP_GROUND,
        GROUP_GROUND,
        GROUP_OTHER,
        GROUP_OTHER,
        GROUP_OTHER,
    ]
