"""Phase 3 losses: confidence-weighted, scale-aligned LiDAR loss and distillation to the zero-shot model.

Both are per-image means averaged over images, as eval averages per-image metrics. Medians use
torch.quantile(x, 0.5), which averages the two middle values like np.median in src/eval.py.
"""

import torch


def _median(x: torch.Tensor) -> torch.Tensor:
    return torch.quantile(x, 0.5)


def lidar_loss(depth, conf, gt, mask, lambda_c: float) -> tuple[torch.Tensor, dict]:
    """mean(c * e - lambda_c * log c) over valid pixels, e = |s * d - g| / g.

    s = median(g / d) per image, detached, the scale the median alignment in eval uses. Images
    without valid pixels are skipped. Returns the loss and the mean per-image AbsRel (detached).
    """
    losses, absrels = [], []
    for d, c, g, m in zip(depth, conf, gt, mask):
        if not m.any():
            continue
        d, c, g = d[m].float(), c[m].float(), g[m].float()
        s = _median(g / d).detach()
        e = (s * d - g).abs() / g
        losses.append((c * e - lambda_c * torch.log(c)).mean())
        absrels.append(e.detach().mean())
    if not losses:
        return depth.sum() * 0.0, {"absrel": float("nan")}
    return torch.stack(losses).mean(), {"absrel": torch.stack(absrels).mean().item()}


def distill_loss(depth, teacher, space: str) -> torch.Tensor:
    """L1 between median-normalised student and teacher over all pixels.

    `linear` compares d / med(d) with t / med(t); `log` compares their logs. Medians are taken
    per image and treated as constants, so the loss is invariant to the scale of either input.
    """
    if space not in ("linear", "log"):
        raise ValueError(f"Unknown distillation space: {space}")
    losses = []
    for d, t in zip(depth, teacher):
        d, t = d.float(), t.float()
        dn = d / _median(d.reshape(-1)).detach()
        tn = t / _median(t.reshape(-1))
        if space == "log":
            dn, tn = torch.log(dn), torch.log(tn)
        losses.append((dn - tn.detach()).abs().mean())
    return torch.stack(losses).mean()
