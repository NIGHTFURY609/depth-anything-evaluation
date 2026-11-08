"""Phase 3 losses: confidence-weighted, scale-aligned LiDAR loss and distillation to the zero-shot model.

Both are per-image means averaged over images, as eval averages per-image metrics. Medians use
torch.quantile(x, 0.5), which averages the two middle values like np.median in src/eval.py.
"""

import torch


def _median(x: torch.Tensor) -> torch.Tensor:
    return torch.quantile(x, 0.5)


def lidar_loss(depth, conf, gt, mask, lambda_c: float, min_valid: int = 1) -> tuple[torch.Tensor, dict]:
    """mean(c * e - lambda_c * log c) over valid pixels, e = |s * d - g| / g.

    s = median(g / d) per image, the scale the median alignment in eval uses. s is NOT detached:
    with it in the graph the loss is exactly scale-invariant, gradient included. Detached, it
    pushed the overall depth scale down every step until it underflowed (DECISIONS.md,
    2026-11-06; scripts/diagnostics/overfit_diverge.py). Images
    with fewer than min_valid valid pixels are skipped, as eval skips them (eval.min_valid).
    Returns the loss and the mean per-image AbsRel (detached).
    """
    losses, absrels = [], []
    for d, c, g, m in zip(depth, conf, gt, mask):
        if int(m.sum()) < max(min_valid, 1):
            continue
        d, c, g = d[m].float(), c[m].float(), g[m].float()
        s = _median(g / d)
        e = (s * d - g).abs() / g
        losses.append((c * e - lambda_c * torch.log(c)).mean())
        absrels.append(e.detach().mean())
    if not losses:
        return depth.sum() * 0.0, {"absrel": float("nan")}
    return torch.stack(losses).mean(), {"absrel": torch.stack(absrels).mean().item()}


def distill_loss(depth, teacher, space: str) -> torch.Tensor:
    """L1 between median-normalised student and teacher over all pixels.

    `linear` compares d / med(d) with t / med(t); `log` compares their logs. Medians are taken
    per image; the student's stays in the graph so the gradient has no scale component.
    """
    if space not in ("linear", "log"):
        raise ValueError(f"Unknown distillation space: {space}")
    losses = []
    for d, t in zip(depth, teacher):
        d, t = d.float(), t.float()
        dn = d / _median(d.reshape(-1))
        tn = t / _median(t.reshape(-1))
        if space == "log":
            dn, tn = torch.log(dn), torch.log(tn)
        losses.append((dn - tn.detach()).abs().mean())
    return torch.stack(losses).mean()
