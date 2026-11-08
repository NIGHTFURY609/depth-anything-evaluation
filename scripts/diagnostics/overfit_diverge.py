"""Phase 3: why does the overfit check diverge? Logs per-step evidence for four hypotheses.

    python scripts/diagnostics/overfit_diverge.py --variant base|noconf|lowlr|fp32 [--steps 150]

base    the overfit check as run (LoRA 1e-4, head 5e-5, constant lr, lambda_c 1)
noconf  confidence term off: loss = mean relative error (tests unbounded confidence weights)
lowlr   rates / 10 (tests plain step size)
fp32    backbone without bf16 autocast (tests bf16 overflow)
Every row logs the loss, the raw predicted depth scale (tests scale drift), confidence
quantiles, the gradient norm before clipping, and train AbsRel. Same 10 frames as the check.
"""
import argparse
import sys

sys.path.insert(0, ".")
import numpy as np
import torch

from scripts.train import build_net, loader
from src.config import load_config, set_seed
from src.infer import eval_frames
from src.losses import lidar_loss
from src.models import forward_da3
from src.train import param_groups
from src.train_data import TrainFrames

parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="base", choices=["base", "noconf", "lowlr", "fp32"])
parser.add_argument("--steps", type=int, default=150)
args = parser.parse_args()

cfg = load_config()
tcfg = cfg["train"]
set_seed(0)
net = build_net(cfg, "lora", 0)
frames = eval_frames(cfg, "train", stride=1)
frames = frames[:: len(frames) // 10][:10]
ds = TrainFrames(cfg, frames, augment=False)
scale = 0.1 if args.variant == "lowlr" else 1.0
opt = torch.optim.AdamW(param_groups(net, 1e-4 * scale, tcfg["lr_head"] * scale), weight_decay=tcfg["weight_decay"])
params = [p for p in net.parameters() if p.requires_grad]


def forward(x):
    if args.variant != "fp32":
        return forward_da3(net, x)
    H, W = x.shape[-2:]
    feats, _ = net.backbone(x[:, None], cam_token=None, export_feat_layers=[])
    out = net.head(feats, H, W, patch_start_idx=0)
    return {"depth": out.depth[:, 0], "conf": out.depth_conf[:, 0]}


print(f"variant {args.variant}")
print("step  loss     absrel  depth_med  depth_min  conf_p50  conf_p99  conf_max  grad_norm")
gen = torch.Generator().manual_seed(0)
step = 0
while step < args.steps:
    for batch in loader(ds, tcfg, gen, workers=2):
        out = forward(batch["image"].cuda())
        gt, mask = batch["gt"].cuda(), batch["mask"].cuda()
        conf = torch.ones_like(out["conf"]) if args.variant == "noconf" else out["conf"]
        loss, stats = lidar_loss(out["depth"], conf, gt, mask, tcfg["lambda_c"], cfg["eval"]["min_valid"])
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"]).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step <= 3 or not np.isfinite(loss.item()):
            d = out["depth"].detach().float()
            c = out["conf"].detach().float()[mask]
            q = torch.quantile(c[torch.randperm(c.numel())[:100000]], torch.tensor([0.5, 0.99], device=c.device))
            print(f"{step:4d}  {loss.item():8.4f} {stats['absrel']:7.4f}  {d.median().item():9.4g}  {d.min().item():9.3g}  "
                  f"{q[0].item():8.2f}  {q[1].item():8.2f}  {c.max().item():8.3g}  {gn:9.3g}", flush=True)
        if not np.isfinite(loss.item()) or step >= args.steps:
            break
    if not np.isfinite(loss.item()):
        print("non-finite loss: stopped")
        break
