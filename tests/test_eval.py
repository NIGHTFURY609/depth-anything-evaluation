import cv2
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.eval import (
    align, auroc, block_bootstrap, budget_remove, edge_band, fit_median_scale,
    fit_scale_shift_huber, fit_scale_shift_ls, paired_block_bootstrap, pixel_metrics,
    sobel_mask, sparsification, subset_masks, valid_mask,
)
from src.rellis import GROUP_GRASS, GROUP_TREES

RNG = np.random.default_rng(0)


def offroadnav_sobel(img, thickness):
    """Verbatim from LARIAD/Offroad-Nav @ 69640cf (MIT), utils.py::edge_detection, sobel branch,
    with the min-max uint8 step from depth_anythingv2.py. Reference for sobel_mask only."""
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    gray = img.astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    maxv = mag.max()
    mag = (mag / maxv) * 255.0 if maxv > 0 else np.zeros_like(mag)
    _, edges = cv2.threshold(mag.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges > 0


# --- valid set and alignment -------------------------------------------------------------

def test_valid_mask_uses_keep_and_depth_range_only():
    depth = np.array([[0.0, 0.5, 5.0, 90.0, 10.0]])
    keep = np.array([[False, True, True, True, False]])
    assert valid_mask(depth, keep, 1.0, 80.0).tolist() == [[False, False, True, False, False]]


def test_median_scale_survives_outliers():
    gt = RNG.uniform(2, 50, 1000)
    pred = gt / 3.0
    pred[:300] *= RNG.uniform(0.2, 5, 300)
    assert fit_median_scale(pred, gt) == pytest.approx(3.0, rel=0.05)


def test_least_squares_recovers_exact_scale_shift():
    x = RNG.uniform(0, 1, 500)
    s, t = fit_scale_shift_ls(x, 2.5 * x - 0.3)
    assert (s, t) == (pytest.approx(2.5), pytest.approx(-0.3))


def test_huber_recovers_scale_shift_with_and_without_outliers():
    x = RNG.uniform(0, 1, 2000)
    y = 2.5 * x - 0.3
    assert fit_scale_shift_huber(x, y) == (pytest.approx(2.5), pytest.approx(-0.3))
    y_noisy = y + RNG.normal(0, 0.01, x.size)
    y_noisy[:200] += 5.0
    s, t = fit_scale_shift_huber(x, y_noisy)
    assert s == pytest.approx(2.5, abs=0.05) and t == pytest.approx(-0.3, abs=0.05)
    _, t_ls = fit_scale_shift_ls(x, y_noisy)
    assert abs(t_ls + 0.3) > 0.3  # a constant offset on 10% of points biases the intercept


def test_median_alignment_of_scaled_depth_is_exact():
    gt = RNG.uniform(2, 50, (8, 8))
    mask = np.ones_like(gt, bool)
    depth, info = align(gt / 4.0, "depth", gt, mask, "median", 1.0, 80.0)
    np.testing.assert_allclose(depth, gt)
    assert info["s"] == pytest.approx(4.0) and info["clamped"] == 0.0


@pytest.mark.parametrize("method", ["ls", "huber"])
def test_scale_shift_alignment_of_affine_disparity_is_exact(method):
    gt = RNG.uniform(2, 50, (8, 8))
    mask = np.ones_like(gt, bool)
    depth, _ = align(3.0 / gt + 0.1, "disparity", gt, mask, method, 1.0, 80.0)
    np.testing.assert_allclose(depth, gt, rtol=1e-6)


def test_scale_shift_on_depth_model_fits_inverse_depth():
    gt = RNG.uniform(2, 50, (8, 8))
    mask = np.ones_like(gt, bool)
    pred_depth = 1.0 / (2.0 / gt + 0.05)  # affine in inverse depth, not a pure scale
    depth, _ = align(pred_depth, "depth", gt, mask, "ls", 1.0, 80.0)
    np.testing.assert_allclose(depth, gt, rtol=1e-6)


def test_nonpositive_disparity_is_clamped_to_max_depth_and_counted():
    gt = np.array([[2.0, 4.0, 8.0, 16.0]])
    mask = np.array([[True, True, True, False]])
    pred = np.array([[0.5, 0.25, 0.125, -1.0]])  # exact for the fit set; last pixel negative
    depth, info = align(pred, "disparity", gt, mask, "ls", 1.0, 80.0)
    assert depth[0, 3] == pytest.approx(80.0)
    assert np.all((depth >= 1.0) & (depth <= 80.0))
    assert info["clamped"] == 0.0  # counted over the valid set only


def test_clamped_fraction_counts_valid_pixels_outside_range():
    gt = np.full((1, 4), 10.0)
    mask = np.ones((1, 4), bool)
    pred = np.array([[1.0, 1.0, 1.0, 100.0]])
    depth, info = align(pred, "depth", gt, mask, "median", 1.0, 80.0)
    assert depth[0, 3] == 80.0 and info["clamped"] == 0.25


# --- metrics and subsets -----------------------------------------------------------------

def test_perfect_prediction_metrics():
    gt = RNG.uniform(2, 50, 100)
    m = pixel_metrics(gt.copy(), gt)
    assert (m["absrel"], m["rmse"], m["delta1"], m["phantom"], m["overshoot"]) == (0, 0, 1, 0, 0)
    assert m["n"] == 100


def test_metric_values_by_hand():
    gt = np.array([10.0, 10.0, 10.0, 10.0])
    pred = np.array([8.0, 7.0, 13.0, 10.0])
    m = pixel_metrics(pred, gt)
    assert m["absrel"] == pytest.approx((0.2 + 0.3 + 0.3 + 0) / 4)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 9 + 9 + 0) / 4))
    assert m["delta1"] == pytest.approx(0.25)  # 10/8 = 1.25 is not < 1.25; 7 and 13 fail too
    assert m["phantom"] == pytest.approx(0.25)  # 8.0 is exactly 0.8g: strict, so only 7.0
    assert m["overshoot"] == pytest.approx(0.25)


def test_empty_subset_gives_nan():
    m = pixel_metrics(np.array([]), np.array([]))
    assert np.isnan(m["absrel"]) and m["n"] == 0


def test_subsets_include_vegetation_union():
    labels = np.array([[3, 4, 19, 1, 0, 7]])
    valid = np.ones_like(labels, bool)
    edges = {"edge_depth": np.zeros_like(valid), "edge_semantic": np.zeros_like(valid)}
    edges["edge_depth"][0, 0] = edges["edge_semantic"][0, 3] = True
    s = subset_masks(valid, labels, edges)
    assert s["all"].sum() == 6
    assert s["vegetation"].tolist() == (s["grass"] | s["trees_bushes"]).tolist()
    assert s["vegetation"].sum() == 3 and s["ground"].sum() == 1
    assert s["edge"].tolist() == [[True, False, False, True, False, False]]


# --- edge band ---------------------------------------------------------------------------

def ring_image(depth_of_row, height=40, width=60, spacing=7):
    """Sparse LiDAR-like GT: one ring every `spacing` rows, every second column."""
    depth = np.zeros((height, width))
    for r in range(3, height, spacing):
        depth[r, ::2] = depth_of_row(r)
    return depth


def test_flat_ground_rings_are_not_edges():
    # Ground: depth falls steeply from ring to ring, but is constant along each ring.
    depth = ring_image(lambda r: 200.0 / (r + 1))
    valid = depth > 0
    labels = np.where(valid, 3, 0)
    edges = edge_band(depth, valid, labels, k=5, tau=0.15)
    assert not edges["edge_depth"].any() and not edges["edge_semantic"].any()


def test_horizontal_depth_step_is_an_edge():
    depth = ring_image(lambda r: 10.0)
    depth[:, 30:] *= 3  # a trunk edge: near on the left, far on the right
    valid = depth > 0
    labels = np.where(valid, 3, 0)
    edges = edge_band(depth, valid, labels, k=5, tau=0.15)
    assert not edges["edge_semantic"].any()
    band = edges["edge_depth"]
    cols = np.nonzero(band.any(axis=0))[0]
    assert band.any() and cols.min() >= 30 - 5 and cols.max() <= 29 + 5


def test_group_boundary_is_an_edge_but_grass_bush_is_not():
    depth = ring_image(lambda r: 10.0)
    valid = depth > 0
    labels = np.where(valid, 3, 0)
    labels[:, 30:] = np.where(valid[:, 30:], 19, 0)  # grass | bush: same merged group
    assert not edge_band(depth, valid, labels, k=5, tau=0.15)["edge_semantic"].any()
    labels[:, 30:] = np.where(valid[:, 30:], 1, 0)  # grass | dirt: a real boundary
    edges = edge_band(depth, valid, labels, k=5, tau=0.15)
    assert edges["edge_semantic"].any() and not edges["edge_depth"].any()


# --- confidence quality ------------------------------------------------------------------

FRACTIONS = np.arange(0, 0.951, 0.05)


def test_oracle_confidence_gives_zero_ause_and_positive_aurg():
    err = RNG.uniform(0, 1, 2000)
    s = sparsification(err, -err, FRACTIONS, np.random.default_rng(0))
    assert s["ause"] == pytest.approx(0.0, abs=1e-12)
    assert s["aurg"] > 0
    assert s["conf"][0] == pytest.approx(err.mean())


def test_constant_confidence_has_no_raster_order_bias():
    # Errors sorted by position; removal in raster order would look like a strong ranking.
    err = np.linspace(1, 0, 4000)
    s = sparsification(err, np.ones_like(err), FRACTIONS, np.random.default_rng(0))
    assert abs(s["aurg"]) < 0.01


def test_auroc_cases_and_sklearn_agreement():
    pos = np.r_[np.ones(30, bool), np.zeros(30, bool)]
    assert auroc(np.r_[np.ones(30), np.zeros(30)], pos) == 1.0
    assert auroc(np.r_[np.zeros(30), np.ones(30)], pos) == 0.0
    assert auroc(np.ones(60), pos) == 0.5
    assert np.isnan(auroc(np.ones(60), np.zeros(60, bool)))
    assert np.isnan(auroc(RNG.random(60), np.r_[np.ones(5, bool), np.zeros(55, bool)]))
    score = np.round(RNG.random(500), 2)  # with ties
    labels = RNG.random(500) < 0.3
    assert auroc(score, labels) == pytest.approx(roc_auc_score(labels, score))


# --- filtering ---------------------------------------------------------------------------

def smooth_scene():
    disp = np.ones((80, 120)) * 0.2
    disp[30:, :] = np.linspace(0.3, 1.0, 50)[:, None]
    disp[10:50, 40:60] = 0.8  # an object
    return disp + RNG.normal(0, 0.002, disp.shape)


@pytest.mark.parametrize("thickness", [3, 5, 9])
def test_sobel_mask_matches_offroadnav(thickness):
    disp = smooth_scene()
    np.testing.assert_array_equal(sobel_mask(disp, thickness), offroadnav_sobel(disp, thickness))


def test_sobel_mask_is_invariant_to_affine_changes():
    disp = smooth_scene()
    np.testing.assert_array_equal(sobel_mask(disp, 5), sobel_mask(7.0 * disp + 3.0, 5))


def test_sobel_mask_of_constant_image_is_empty():
    assert not sobel_mask(np.ones((20, 30)), 5).any()


def test_budget_removes_exactly_n_lowest_scored_candidates():
    scores = np.array([[5.0, 1.0, 3.0], [2.0, 4.0, 0.0]])
    candidates = np.array([[True, True, True], [True, True, False]])
    removed = budget_remove(scores, 2, candidates, np.random.default_rng(0))
    assert removed.tolist() == [[False, True, False], [True, False, False]]


def test_budget_breaks_ties_randomly():
    scores = np.ones((10, 100))
    removed = budget_remove(scores, 500, np.ones_like(scores, bool), np.random.default_rng(0))
    assert removed.sum() == 500
    assert 0.3 < removed[:5].mean() < 0.7  # not the first rows in raster order


# --- bootstrap ---------------------------------------------------------------------------

def test_bootstrap_of_constant_has_zero_width():
    mean, lo, hi = block_bootstrap(np.full(95, 0.3), block=10, resamples=200, seed=0)
    assert mean == lo == hi == pytest.approx(0.3)


def test_bootstrap_is_seeded_and_ignores_nan():
    v = RNG.random(100)
    v[::7] = np.nan
    assert block_bootstrap(v, 10, 300, 1) == block_bootstrap(v, 10, 300, 1)
    assert block_bootstrap(v, 10, 300, 1)[0] == pytest.approx(np.nanmean(v))


def test_paired_bootstrap_of_identical_inputs_is_zero():
    v = RNG.random(100)
    assert paired_block_bootstrap(v, v.copy(), 10, 200, 0) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_uses_only_images_valid_for_both():
    a = np.array([1.0, 2.0, np.nan, 4.0] * 10)
    b = np.array([0.0, np.nan, 3.0, 3.0] * 10)
    mean, lo, hi = paired_block_bootstrap(a, b, 4, 100, 0)
    assert mean == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_jump_partner_need_not_be_scored_itself():
    """The occlusion filter drops far points next to near ones, so edges are found among all
    LiDAR points and only then restricted to the valid set."""
    depth = ring_image(lambda r: 10.0)
    depth[:, 30:] *= 3
    points = depth > 0
    valid = points.copy()
    valid[:, 30:36] = False  # far points the occlusion filter removed
    labels = np.where(points, 3, 0)
    band = edge_band(depth, points, labels, k=5, tau=0.15)["edge_depth"] & valid
    assert band[:, 26:30].any()


# --- per-frame evaluation and the freeze gate --------------------------------------------

from src.eval import check_can_evaluate, evaluate_frame, sobel_score  # noqa: E402

ECFG = {
    "min_depth": 1.0, "max_depth": 80.0, "min_valid": 200, "min_subset_px": 100, "edge_k": 4,
    "edge_tau": 0.2, "distance_bins": [0, 10, 20, 80], "high_error": 0.25, "auroc_min_class": 20,
    "phantom_ratio": 0.8, "overshoot_ratio": 1.25, "sparsification_step": 0.05,
    "sparsification_max": 0.95, "sobel_thickness": [3, 5], "random_filter_seeds": 3,
    "filter_budgets": [0.1, 0.5], "sobel_rank_thickness": 5,
}


def synthetic_gt(annotated=False):
    depth = ring_image(lambda r: 5.0 + r, height=60, width=80, spacing=3)
    labels = np.where(depth > 0, 3, 0).astype(np.uint8)
    labels[:, 40:] = np.where(depth[:, 40:] > 0, 4, 0)
    gt = {"depth": depth, "labels": labels, "keep": depth > 0}
    if annotated:
        gt["image_labels"] = np.full_like(labels, 3)
    return gt


def dense(depth):
    """Fills every row with its ring's depth, so a prediction exists at every pixel."""
    out = depth.copy()
    for r in range(out.shape[0]):
        ring = depth[max(r - 1, 0) : r + 2].max()
        out[r] = ring if ring > 0 else 50.0
    return out


def test_perfect_depth_prediction_scores_perfectly():
    gt = synthetic_gt(annotated=True)
    pred = dense(gt["depth"]) / 3.0
    conf = np.ones_like(pred)
    out = evaluate_frame(pred, conf, "depth", gt, ECFG, np.random.default_rng(0))
    acc = {(r["align"], r["label_source"], r["subset"]): r for r in out["accuracy"]}
    for align_method in ("median", "ls", "huber"):
        row = acc[(align_method, "lidar", "all")]
        assert row["absrel"] == pytest.approx(0, abs=1e-9) and row["delta1"] == 1.0
    assert ("median", "image", "grass") in acc
    assert ("median", "lidar", "grass|0-10m") in acc
    assert {r["align"] for r in out["confidence"]} == {"median", "ls", "huber"}
    filters = {r["filter"] for r in out["filtering"]}
    assert {"none", "sobel3", "conf_valid3", "conf_all3", "random_valid3", "sobel5"} <= filters
    assert {"sobelrank0.1", "confrank0.1", "randomrank0.1", "confrank0.5"} <= filters


def test_disparity_model_has_no_confidence_rows_or_median():
    gt = synthetic_gt()
    pred = 1.0 / dense(gt["depth"])
    out = evaluate_frame(pred, None, "disparity", gt, ECFG, np.random.default_rng(0))
    assert out["confidence"] == []
    assert {r["align"] for r in out["accuracy"]} == {"ls", "huber"}
    assert not any(r["filter"].startswith("conf") for r in out["filtering"])


def test_small_subsets_are_nan_but_counted():
    gt = synthetic_gt()
    out = evaluate_frame(dense(gt["depth"]), np.ones((60, 80)), "depth", gt, ECFG, np.random.default_rng(0))
    ground = [r for r in out["accuracy"] if r["subset"] == "ground" and r["align"] == "median"][0]
    assert ground["n"] == 0 and np.isnan(ground["absrel"])


def test_too_few_valid_pixels_skips_the_frame():
    gt = synthetic_gt()
    gt["keep"][:] = False
    gt["keep"][3, :10:2] = True
    out = evaluate_frame(dense(gt["depth"]), None, "depth", gt, ECFG, np.random.default_rng(0))
    assert out["accuracy"] == [] and out["skipped"]


def test_test_split_requires_frozen_config_and_clean_tree():
    check_can_evaluate("val", frozen=False, dirty=True)
    with pytest.raises(SystemExit):
        check_can_evaluate("test", frozen=False, dirty=False)
    with pytest.raises(SystemExit):
        check_can_evaluate("test", frozen=True, dirty=True)
    check_can_evaluate("test", frozen=True, dirty=False)


def test_removal_curve_matches_brute_force():
    err = RNG.uniform(0, 1, 137)
    conf = np.round(RNG.uniform(1, 3, 137), 1)  # many ties
    s = sparsification(err, conf, FRACTIONS, np.random.default_rng(5))
    order = np.lexsort((np.random.default_rng(5).random(137), conf))
    for f, value in zip(FRACTIONS, s["conf"]):
        k = int(np.floor(f * 137))
        assert value == pytest.approx(err[order[k:]].mean())
    by_err = np.sort(err)
    for f, value in zip(FRACTIONS, s["oracle"]):
        assert value == pytest.approx(by_err[: 137 - int(np.floor(f * 137))].mean())


def filtering_rows(out, align="median"):
    return {r["filter"]: r for r in out["filtering"] if r["align"] == align}


def test_filters_in_a_group_remove_the_same_number_of_valid_pixels():
    gt = synthetic_gt()
    pred = dense(gt["depth"]) * RNG.uniform(0.9, 1.1, (60, 80))
    conf = RNG.uniform(1, 5, (60, 80))
    rows = filtering_rows(evaluate_frame(pred, conf, "depth", gt, ECFG, np.random.default_rng(0)))
    n_valid = int((gt["keep"]).sum())
    for th in (3, 5):
        removed = {rows[f"{k}{th}"]["removed_valid"] for k in ("sobel", "conf_valid", "random_valid")}
        assert max(removed) - min(removed) < 1e-12
    for b in (0.1, 0.5):
        for k in ("sobelrank", "confrank", "randomrank"):
            assert rows[f"{k}{b:g}"]["removed_valid"] == pytest.approx(round(b * n_valid) / n_valid)


def test_a_filter_leaving_too_few_pixels_voids_its_whole_group():
    gt = synthetic_gt()
    pred = dense(gt["depth"])
    cfg = {**ECFG, "filter_budgets": [0.95], "min_subset_px": 50}  # 5% of 760 = 38 left
    rows = filtering_rows(evaluate_frame(pred, np.ones((60, 80)), "depth", gt, cfg, np.random.default_rng(0)))
    assert all(np.isnan(rows[k]["absrel"]) for k in ("sobelrank0.95", "confrank0.95", "randomrank0.95"))
    assert np.isfinite(rows["none"]["absrel"])


def test_evaluate_frame_is_deterministic_for_a_seed():
    gt = synthetic_gt(annotated=True)
    pred = dense(gt["depth"]) * RNG.uniform(0.8, 1.2, (60, 80))
    conf = np.round(RNG.uniform(1, 2, (60, 80)), 1)
    a = evaluate_frame(pred, conf, "depth", gt, ECFG, np.random.default_rng(7))
    b = evaluate_frame(pred, conf, "depth", gt, ECFG, np.random.default_rng(7))
    for key in ("accuracy", "confidence", "filtering"):
        assert str(a[key]) == str(b[key])


def test_sobel_score_ranks_the_masked_pixels_first():
    disp = smooth_scene()
    score = sobel_score(disp, 5)
    mask = sobel_mask(disp, 5)
    assert score[mask].min() >= np.median(score[~mask])


def test_bootstrap_blocks_follow_frame_positions():
    """Values on every other frame: positions put 15 of them in each block of 30 frames."""
    v = RNG.random(60)
    by_position = block_bootstrap(v, 30, 300, 0, positions=np.arange(60) * 2)
    by_row = block_bootstrap(v, 15, 300, 0)
    assert by_position == by_row
