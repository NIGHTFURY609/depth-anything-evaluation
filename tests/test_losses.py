import math

import pytest
import torch

from src.losses import distill_loss, lidar_loss


def batch(seed=0, b=2, h=6, w=8):
    g = torch.Generator().manual_seed(seed)
    depth = torch.rand(b, h, w, generator=g) * 10 + 1
    gt = depth * (1 + 0.3 * (torch.rand(b, h, w, generator=g) - 0.5)) * 2.5
    conf = 1 + torch.rand(b, h, w, generator=g) * 3
    mask = torch.rand(b, h, w, generator=g) > 0.3
    return depth, conf, gt, mask


def test_lidar_loss_ignores_prediction_scale():
    depth, conf, gt, mask = batch()
    a, _ = lidar_loss(depth, conf, gt, mask, 1.0)
    b, _ = lidar_loss(depth * 7.3, conf, gt, mask, 1.0)
    assert torch.allclose(a, b, atol=1e-5)


def test_lidar_loss_matches_hand_computation():
    depth, conf, gt, mask = batch(b=1)
    loss, stats = lidar_loss(depth, conf, gt, mask, 0.7)
    d, c, g = depth[mask], conf[mask], gt[mask]
    s = torch.quantile(g / d, 0.5)
    e = (s * d - g).abs() / g
    assert torch.allclose(loss, (c * e - 0.7 * torch.log(c)).mean(), atol=1e-6)
    assert math.isclose(stats["absrel"], e.mean().item(), rel_tol=1e-5)


def test_scale_is_detached():
    """The gradient is that of c*|s*d - g|/g with s held fixed."""
    depth, conf, gt, mask = batch(b=1)
    depth.requires_grad_(True)
    loss, _ = lidar_loss(depth, conf, gt, mask, 1.0)
    loss.backward()
    with torch.no_grad():
        s = torch.quantile(gt[mask] / depth[mask], 0.5)
        expected = torch.zeros_like(depth)
        expected[mask] = conf[mask] * s * torch.sign(s * depth[mask] - gt[mask]) / gt[mask] / mask.sum()
    assert torch.allclose(depth.grad, expected, atol=1e-6)


def test_masked_pixels_have_no_effect():
    depth, conf, gt, mask = batch()
    a, _ = lidar_loss(depth, conf, gt, mask, 1.0)
    depth2, gt2, conf2 = depth.clone(), gt.clone(), conf.clone()
    depth2[~mask], gt2[~mask], conf2[~mask] = 99.0, 0.5, 50.0
    b, _ = lidar_loss(depth2, conf2, gt2, mask, 1.0)
    assert torch.allclose(a, b)


def test_images_are_weighted_equally():
    """Per-image means, averaged over images, as eval averages per-image metrics."""
    depth, conf, gt, mask = batch(b=2)
    mask[1] = False
    mask[1, 0, :2] = True  # 2 pixels in image 1, many in image 0
    both, _ = lidar_loss(depth, conf, gt, mask, 1.0)
    first, _ = lidar_loss(depth[:1], conf[:1], gt[:1], mask[:1], 1.0)
    second, _ = lidar_loss(depth[1:], conf[1:], gt[1:], mask[1:], 1.0)
    assert torch.allclose(both, (first + second) / 2, atol=1e-6)


def test_empty_image_is_skipped_and_empty_batch_is_zero():
    depth, conf, gt, mask = batch(b=2)
    mask[1] = False
    both, _ = lidar_loss(depth, conf, gt, mask, 1.0)
    first, _ = lidar_loss(depth[:1], conf[:1], gt[:1], mask[:1], 1.0)
    assert torch.allclose(both, first)
    depth.requires_grad_(True)
    none, stats = lidar_loss(depth, conf, gt, torch.zeros_like(mask), 1.0)
    none.backward()  # must still be differentiable
    assert none.item() == 0.0 and math.isnan(stats["absrel"])


@pytest.mark.parametrize("err", [0.05, 0.2, 0.5, 2.0])
def test_confidence_optimum_is_lambda_over_error(err):
    """With c >= 1 (DA3's exp(x) + 1), the per-pixel optimum is max(1, lambda_c / e)."""
    gt = torch.ones(1, 1, 3)
    depth = torch.tensor([[[1.0, 1.0, 1.0 + err]]])  # median scale 1, error err on the last pixel
    mask = torch.ones_like(gt, dtype=torch.bool)
    cs = torch.linspace(1, 30, 2901)
    losses = []
    for c in cs:
        conf = torch.tensor([[[1.0, 1.0, c.item()]]])
        losses.append(lidar_loss(depth, conf, gt, mask, 1.0)[0].item())
    best = cs[int(torch.tensor(losses).argmin())].item()
    assert best == pytest.approx(max(1.0, 1.0 / err), abs=0.02)


@pytest.mark.parametrize("space", ["linear", "log"])
def test_distill_is_zero_for_the_teacher_and_ignores_scale(space):
    depth, _, teacher, _ = batch()
    assert distill_loss(teacher, teacher, space).item() == pytest.approx(0, abs=1e-7)
    assert distill_loss(teacher * 3.0, teacher, space).item() == pytest.approx(0, abs=1e-6)
    a = distill_loss(depth, teacher, space)
    assert torch.allclose(a, distill_loss(depth * 0.2, teacher * 9.0, space), atol=1e-5)


def test_distill_values():
    teacher = torch.tensor([[[1.0, 2.0, 4.0]]])  # median 2: normalised 0.5, 1, 2
    student = torch.tensor([[[1.0, 1.0, 1.0]]])  # normalised 1, 1, 1
    assert distill_loss(student, teacher, "linear").item() == pytest.approx((0.5 + 0 + 1) / 3)
    assert distill_loss(student, teacher, "log").item() == pytest.approx(2 * math.log(2) / 3)
    with pytest.raises(ValueError):
        distill_loss(student, teacher, "sqrt")


def test_distill_gradient_treats_medians_as_constants():
    teacher = torch.tensor([[[1.0, 2.0, 4.0]]])
    student = torch.tensor([[[1.0, 3.0, 5.0]]], requires_grad=True)
    distill_loss(student, teacher, "linear").backward()
    # median 3, so d/dx |x/3 - t/2| / 3 = sign(...) / 9
    expected = torch.tensor([[[1.0, 1.0, 1.0]]]) / 9 * torch.sign(student.detach() / 3 - teacher / 2)
    assert torch.allclose(student.grad, expected)
