import numpy as np
import pytest

from src.calib_refine import (
    block_split,
    bootstrap_offsets,
    decide,
    estimate_offset,
    frame_counts,
    precision_at,
    rotation_for_pixel_shift,
)
from src.gt import Calibration, load_calibration
from src.projection import project_points

K = np.array([[2800.0, 0, 960.0], [0, 2800.0, 600.0], [0, 0, 1]])
FULL = (1920, 1200)
TARGET = (504, 322)
RADIUS = 8
CLASSES = [27, 8]


def scene(offset_x=0, offset_y=0, seed=0):
    """A barrier (27) in image labels, with LiDAR points that land shifted by the offset."""
    image = np.full((322, 504), 3, dtype=np.uint8)
    image[100:140, 200:300] = 27
    rng = np.random.default_rng(seed)
    labels = np.zeros((322, 504), dtype=np.uint8)
    rows = rng.integers(100, 140, 80) - offset_y
    cols = rng.integers(200, 300, 80) - offset_x
    labels[rows, cols] = 27
    labels[np.arange(0, 320, 9), 10] = 3  # Grass points must not affect the fit.
    return labels, labels > 0, image


def counts_for(offset_x, offset_y, frames=6):
    grids = [
        frame_counts(*scene(offset_x, offset_y, seed), CLASSES, RADIUS) for seed in range(frames)
    ]
    return np.stack([g[0] for g in grids]), np.stack([g[1] for g in grids])


def test_estimate_recovers_a_known_offset():
    hits, totals = counts_for(3, 5)
    dx, dy = estimate_offset(hits.sum(0), totals.sum(0), RADIUS)
    assert dx == pytest.approx(3, abs=1)
    assert dy == pytest.approx(5, abs=1)


def test_estimate_is_near_zero_when_aligned():
    hits, totals = counts_for(0, 0)
    dx, dy = estimate_offset(hits.sum(0), totals.sum(0), RADIUS)
    assert abs(dx) <= 1 and abs(dy) <= 1


def test_points_of_other_classes_are_ignored():
    labels, keep, image = scene()
    hits, totals = frame_counts(labels, keep, image, [8], RADIUS)
    assert hits.sum() == 0 and totals.sum() == 0


def test_occlusion_filtered_points_are_ignored():
    labels, _, image = scene()
    _, totals = frame_counts(labels, np.zeros_like(labels, dtype=bool), image, CLASSES, RADIUS)
    assert totals.sum() == 0


def test_precision_at_zero_and_true_offset():
    hits, totals = counts_for(4, 0)
    assert precision_at(hits.sum(0), totals.sum(0), RADIUS, 4, 0) > precision_at(
        hits.sum(0), totals.sum(0), RADIUS, 0, 0
    )


def test_block_split_keeps_neighbouring_frames_together():
    a, b = block_split(100, block=10)
    assert sorted(np.concatenate([a, b]).tolist()) == list(range(100))
    assert 0 in a and 9 in a and 10 in b and 19 in b


def test_bootstrap_is_tight_with_consistent_frames():
    hits, totals = counts_for(3, 5, frames=20)
    low, high = bootstrap_offsets(hits, totals, RADIUS, block=2, resamples=50, seed=0)
    assert low[1] <= 5 + 1 and high[1] >= 5 - 1
    assert high[1] - low[1] <= 2


def test_rotation_moves_points_by_the_requested_pixels():
    xyz = np.array([[1.0, 0.5, 40.0], [-3.0, 1.0, 15.0], [0.0, -1.0, 8.0]])
    base = project_points(xyz, np.eye(4), K, np.zeros(5), FULL, TARGET)
    T = rotation_for_pixel_shift(3.0, 5.0, K, FULL, TARGET)
    moved = project_points(xyz, T, K, np.zeros(5), FULL, TARGET)
    # A rotation is not exactly a uniform shift: off-axis it scales by 1 + tan^2(angle), about
    # 1.6% for the 7 degree point here, so 5 px becomes 5.08 px.
    assert moved[0] - base[0] == pytest.approx([3.0, 3.0, 3.0], abs=0.1)
    assert moved[1] - base[1] == pytest.approx([5.0, 5.0, 5.0], abs=0.1)


def test_decide_accepts_a_tight_improving_estimate():
    row = decide(
        dx=2.0, dy=5.0, ci_low=(1.0, 4.0), ci_high=(3.0, 6.0),
        holdout_zero=0.70, holdout_shifted=0.80, points=5000, max_ci_width=3.0, min_points=2000,
    )
    assert row["accepted"] is True
    assert row["reason"] == "ok"


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"ci_low": (0.0, 0.0), "ci_high": (8.0, 9.0)}, "unstable"),
        ({"holdout_shifted": 0.69}, "no_holdout_gain"),
        ({"points": 100}, "too_few_points"),
    ],
)
def test_decide_rejects_weak_estimates(overrides, reason):
    args = dict(
        dx=2.0, dy=5.0, ci_low=(1.0, 4.0), ci_high=(3.0, 6.0),
        holdout_zero=0.70, holdout_shifted=0.80, points=5000, max_ci_width=3.0, min_points=2000,
    )
    row = decide(**{**args, **overrides})
    assert row["accepted"] is False
    assert row["reason"] == reason


def make_calib_tree(tmp_path, seq="00001"):
    intr = tmp_path / "calib/intrinsics/Rellis-3D" / seq
    extr = tmp_path / "calib/basler_to_ouster/Rellis_3D" / seq
    intr.mkdir(parents=True)
    extr.mkdir(parents=True)
    (intr / "camera_info.txt").write_text("2800 2800 960 600\n")
    (extr / "transforms.yaml").write_text(
        "os1_cloud_node-pylon_camera_node:\n"
        "  q: {w: 1, x: 0, y: 0, z: 0}\n"
        "  t: {x: 0, y: 0, z: 0}\n"
    )
    return {
        "paths": {"data_root": str(tmp_path)},
        "input": {"width": 504, "height": 322},
        "gt": {"distortion": [0, 0, 0, 0, 0], "source_size": [1920, 1200]},
    }


def test_load_calibration_is_official_by_default(tmp_path):
    cfg = make_calib_tree(tmp_path)
    assert np.allclose(load_calibration(cfg, "00001").T_cam_lidar, np.eye(4))


def test_load_calibration_applies_corrections_when_given(tmp_path):
    cfg = make_calib_tree(tmp_path)
    path = tmp_path / "estimates.yaml"
    path.write_text("'00001': {dx: 0.0, dy: 4.0}\n")
    corrected: Calibration = load_calibration(cfg, "00001", corrections=path)
    assert not np.allclose(corrected.T_cam_lidar, np.eye(4))


def test_load_calibration_requires_an_entry_for_the_sequence(tmp_path):
    cfg = make_calib_tree(tmp_path)
    path = tmp_path / "estimates.yaml"
    path.write_text("'00002': {dx: 1.0, dy: 1.0}\n")
    with pytest.raises(KeyError):
        load_calibration(cfg, "00001", corrections=path)
