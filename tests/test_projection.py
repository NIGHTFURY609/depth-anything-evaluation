import cv2
import numpy as np
import pytest

from src.projection import (
    label_agreement,
    occlusion_mask,
    project_points,
    shift_agreement,
    zbuffer,
)

K = np.array([[2800.0, 0, 960.0], [0, 2800.0, 600.0], [0, 0, 1]])
NO_DIST = np.zeros(5)
DIST = np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0.0])
FULL = (1920, 1200)
TARGET = (504, 322)
SX, SY = 504 / 1920, 322 / 1200


def project(xyz, dist=NO_DIST, T=np.eye(4), dst=TARGET):
    return project_points(np.array(xyz, dtype=np.float64), T, K, dist, FULL, dst)


def test_axis_point_lands_on_scaled_principal_point():
    u, v, z, _ = project([[0, 0, 10]])
    # Pixel centres sit at integers, so scaling goes through the half-pixel offset.
    assert u[0] == pytest.approx((960 + 0.5) * SX - 0.5)
    assert v[0] == pytest.approx((600 + 0.5) * SY - 0.5)
    assert z[0] == pytest.approx(10)


def test_identity_scale_keeps_full_resolution_pixels():
    u, v, _, _ = project([[1, 0, 10]], dst=FULL)
    assert u[0] == pytest.approx(960 + 2800 * 0.1)
    assert v[0] == pytest.approx(600)


def test_depth_is_camera_z_not_range():
    _, _, z, _ = project([[1, 0, 4]])
    assert z[0] == pytest.approx(4)


def test_points_behind_camera_are_dropped():
    u, _, _, kept = project([[0, 0, -5], [0, 0, 5]])
    assert len(u) == 1
    assert kept.tolist() == [1]


def test_extrinsics_are_applied():
    T = np.eye(4)
    T[2, 3] = 5.0
    _, _, z, _ = project([[0, 0, 10]], T=T)
    assert z[0] == pytest.approx(15)


def test_distortion_matches_opencv():
    xyz = np.array([[1.0, 0.5, 10.0]])
    u, v, _, _ = project(xyz, dist=DIST, dst=FULL)
    expected, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, DIST)
    assert u[0] == pytest.approx(expected[0, 0, 0])
    assert v[0] == pytest.approx(expected[0, 0, 1])


def test_far_off_axis_point_cannot_fold_back_into_the_image():
    # The distortion polynomial is not monotonic: at x/z = 5 it maps back towards the image.
    u, _, _, _ = project([[50, 0, 10]], dist=DIST, dst=FULL)
    assert len(u) == 0


def test_kept_index_aligns_with_output():
    _, _, z, kept = project([[0, 0, -1], [0, 0, 7], [50, 0, 10], [0, 0, 9]])
    assert kept.tolist() == [1, 3]
    assert z.tolist() == pytest.approx([7, 9])


def test_zbuffer_nearest_wins_and_keeps_its_label():
    depth, labels = zbuffer(
        np.array([2.0, 2.2]), np.array([3.0, 3.1]), np.array([9.0, 4.0]), np.array([1, 2]), 5, 5
    )
    assert depth[3, 2] == pytest.approx(4.0)
    assert labels[3, 2] == 2
    assert np.count_nonzero(depth) == 1


def test_zbuffer_rounds_to_nearest_pixel_and_drops_outside():
    depth, _ = zbuffer(
        np.array([0.4, 4.6, -0.6, 2.0]),
        np.array([0.0, 0.0, 0.0, 4.6]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.zeros(4, dtype=np.uint8),
        5,
        5,
    )
    assert depth[0, 0] == pytest.approx(1.0)
    assert np.count_nonzero(depth) == 1


def sparse(values: dict, shape=(9, 9)) -> np.ndarray:
    depth = np.zeros(shape, dtype=np.float32)
    for (row, col), value in values.items():
        depth[row, col] = value
    return depth


def occlusion(depth):
    return occlusion_mask(depth, radius_x=2, radius_y=2, rel_threshold=0.2, abs_threshold=0.5)


def test_background_beside_foreground_is_removed():
    keep = occlusion(sparse({(4, 4): 5.0, (4, 5): 20.0}))
    assert keep[4, 4]
    assert not keep[4, 5]


def test_background_below_foreground_is_removed():
    keep = occlusion(sparse({(3, 4): 5.0, (4, 4): 20.0}))
    assert not keep[4, 4]


def test_ground_receding_upwards_is_kept():
    # On a ground plane the closer neighbour lies below: that is geometry, not occlusion.
    keep = occlusion(sparse({(5, 4): 5.0, (4, 4): 20.0}))
    assert keep[4, 4] and keep[5, 4]


def test_small_depth_step_is_kept():
    keep = occlusion(sparse({(4, 4): 10.0, (4, 5): 11.0}))
    assert keep[4, 5]


def test_small_absolute_step_is_kept_when_close():
    keep = occlusion(sparse({(4, 4): 1.0, (4, 5): 1.4}))
    assert keep[4, 5]


def test_point_outside_the_window_does_not_occlude():
    keep = occlusion(sparse({(4, 1): 5.0, (4, 5): 20.0}))
    assert keep[4, 5]


def test_empty_pixels_are_never_kept():
    keep = occlusion(sparse({(4, 4): 5.0}))
    assert keep.sum() == 1


def test_label_agreement_ignores_void_and_empty():
    lidar = np.array([[3, 3, 0, 4]], dtype=np.uint8)
    image = np.array([[3, 4, 3, 0]], dtype=np.uint8)
    # Only the first two pixels have labels on both sides, and one of them agrees.
    assert label_agreement(lidar, image) == pytest.approx(0.5)


def test_label_agreement_is_nan_without_overlap():
    assert np.isnan(label_agreement(np.zeros((2, 2), np.uint8), np.ones((2, 2), np.uint8)))


def blocks(shape=(60, 80), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(1, 5, (6, 8), dtype=np.uint8)
    return np.kron(coarse, np.ones((10, 10), dtype=np.uint8))[: shape[0], : shape[1]]


def test_shift_grid_peaks_at_true_offset():
    image = blocks()
    # LiDAR labels that land 3 px right of and 2 px above where they should.
    lidar = np.roll(np.roll(image, 3, axis=1), -2, axis=0)
    lidar[np.random.default_rng(1).random(lidar.shape) < 0.5] = 0
    grid = shift_agreement(lidar, image, radius=5)
    row, col = np.unravel_index(np.nanargmax(grid), grid.shape)
    assert (col - 5, row - 5) == (-3, 2)


def test_shift_grid_peaks_at_zero_when_aligned():
    image = blocks()
    grid = shift_agreement(image, image, radius=5)
    assert np.unravel_index(np.nanargmax(grid), grid.shape) == (5, 5)
