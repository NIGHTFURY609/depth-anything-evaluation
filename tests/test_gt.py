import cv2
import numpy as np
import pytest

from src.gt import Calibration, build_frame, load_calibration, save_frame
from src.rellis import Frame

CFG = {
    "input": {"width": 504, "height": 322},
    "gt": {
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "source_size": [1920, 1200],
        "occlusion": {"radius_x": 4, "radius_y": 4, "rel_threshold": 0.2, "abs_threshold": 0.5},
    },
}
CALIB = Calibration(
    K=np.array([[2800.0, 0, 960.0], [0, 2800.0, 600.0], [0, 0, 1]]),
    T_cam_lidar=np.eye(4),
    dist=np.zeros(5),
)


def make_frame(tmp_path, points, labels, annotated=True) -> Frame:
    scan, point_labels = tmp_path / "000001.bin", tmp_path / "000001.label"
    np.array(points, dtype=np.float32).tofile(scan)
    np.array(labels, dtype=np.uint32).tofile(point_labels)
    image_labels = None
    if annotated:
        image_labels = tmp_path / "frame.png"
        annotation = np.full((1200, 1920), 3, dtype=np.uint8)
        annotation[:, 960:] = 4
        cv2.imwrite(str(image_labels), annotation)
    return Frame("00000", 1, tmp_path / "unused.jpg", scan, point_labels, image_labels)


def test_build_frame_rasterises_depth_labels_at_target_size(tmp_path):
    frame = make_frame(tmp_path, [[0, 0, 10, 0.5]], [19])
    gt = build_frame(frame, CALIB, CFG)
    assert gt["depth"].shape == (322, 504)
    assert np.count_nonzero(gt["depth"]) == 1
    row, col = np.argwhere(gt["depth"] > 0)[0]
    assert gt["depth"][row, col] == pytest.approx(10)
    assert gt["labels"][row, col] == 19


def test_build_frame_stores_filtered_and_unfiltered_points(tmp_path):
    # A far point right beside a near one, two pixels apart at the target size.
    frame = make_frame(tmp_path, [[0, 0, 10, 0.5], [0.15, 0, 50, 0.5]], [3, 3])
    gt = build_frame(frame, CALIB, CFG)
    assert np.count_nonzero(gt["depth"]) == 2
    assert gt["keep"].sum() == 1
    assert gt["depth"][gt["keep"]][0] == pytest.approx(10)


def test_build_frame_resizes_image_labels_without_inventing_classes(tmp_path):
    frame = make_frame(tmp_path, [[0, 0, 10, 0.5]], [3])
    gt = build_frame(frame, CALIB, CFG)
    assert gt["image_labels"].shape == (322, 504)
    assert set(np.unique(gt["image_labels"])) == {3, 4}


def test_build_frame_without_annotation_has_no_image_labels(tmp_path):
    frame = make_frame(tmp_path, [[0, 0, 10, 0.5]], [3], annotated=False)
    assert "image_labels" not in build_frame(frame, CALIB, CFG)


def test_save_frame_round_trips(tmp_path):
    frame = make_frame(tmp_path, [[0, 0, 10, 0.5]], [3])
    gt = build_frame(frame, CALIB, CFG)
    path = save_frame(gt, frame, tmp_path / "cache")
    assert path == tmp_path / "cache" / "gt" / "00000" / "000001.npz"
    with np.load(path) as saved:
        assert np.array_equal(saved["depth"], gt["depth"])
        assert saved["keep"].dtype == bool


def test_save_frame_can_use_a_separate_cache(tmp_path):
    frame = make_frame(tmp_path, [[0, 0, 10, 0.5]], [3])
    gt = build_frame(frame, CALIB, CFG)
    path = save_frame(gt, frame, tmp_path / "cache", subdir="gt_sensitivity")
    assert path == tmp_path / "cache" / "gt_sensitivity" / "00000" / "000001.npz"


def test_load_calibration_uses_the_sequence_files(tmp_path):
    for seq, tz in (("00000", 1), ("00002", 2)):
        intr = tmp_path / "calib/intrinsics/Rellis-3D" / seq
        extr = tmp_path / "calib/basler_to_ouster/Rellis_3D" / seq
        intr.mkdir(parents=True)
        extr.mkdir(parents=True)
        (intr / "camera_info.txt").write_text("10 20 3 4\n")
        (extr / "transforms.yaml").write_text(
            "os1_cloud_node-pylon_camera_node:\n"
            "  q: {w: 1, x: 0, y: 0, z: 0}\n"
            f"  t: {{x: 0, y: 0, z: {tz}}}\n"
        )
    cfg = {"paths": {"data_root": str(tmp_path)}, "gt": {"distortion": [1, 2, 3, 4, 5]}}
    calib = load_calibration(cfg, "00002")
    assert calib.K[0, 0] == 10
    assert calib.T_cam_lidar[2, 3] == pytest.approx(-2)
    assert calib.dist.tolist() == [1, 2, 3, 4, 5]
