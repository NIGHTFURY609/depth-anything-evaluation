"""Trains one Phase 3 run, or runs the overfit sanity check.

    python scripts/train.py --mode lora --lr 1e-4 --lambda-d 0.5 --seed 0
    python scripts/train.py --mode lora --lr 1e-4 --lambda-d 0.5 --epochs 1      # sanity epoch
    python scripts/train.py --overfit 10 --steps 300                             # overfit check

A run resumes from <checkpoint_root>/<name>/last.pt and is skipped once results/phase3/runs/<name>/
done.json exists. Logs: results/phase3/runs/<name>/log.csv and TensorBoard runs/<name>.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import git_hash, load_config, set_seed
from src.eval import align
from src.gt import frame_path
from src.infer import eval_frames
from src.losses import lidar_loss
from src.models import forward_da3, load_da3, set_trainable
from src.train import fit, param_groups, run_name
from src.train_data import TrainFrames, load_resized
from src.val import Validator, frame_masks, frame_metrics, mean_metrics

DEVICE = "cuda"
# A run resumes only if these files are byte-identical to when it started.
CODE_FILES = ["src/losses.py", "src/train.py", "src/train_data.py", "src/val.py", "src/models.py",
              "src/eval.py", "scripts/train.py"]


def code_sha256() -> str:
    h = hashlib.sha256()
    for path in CODE_FILES:
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def build_net(cfg: dict, mode: str, seed: int):
    net = load_da3(cfg["models"][cfg["train"]["base_model"]], DEVICE).model
    torch.manual_seed(seed)  # LoRA's A matrices are initialised randomly
    set_trainable(net, mode, cfg["lora"]["rank"], cfg["lora"]["target_regex"])
    return net


def loader(ds, tcfg: dict, generator, shuffle: bool = True, workers: int | None = None):
    workers = tcfg["num_workers"] if workers is None else workers
    return DataLoader(ds, batch_size=tcfg["batch"], shuffle=shuffle, generator=generator, drop_last=True,
                      num_workers=workers, pin_memory=True, persistent_workers=False)


def train_run(cfg: dict, args) -> dict:
    tcfg = {**cfg["train"], "min_valid": cfg["eval"]["min_valid"]}
    spec = {"mode": args.mode, "lambda_d": args.lambda_d, "seed": args.seed}
    if args.mode != "head":
        spec["lr"] = args.lr
    spec["name"] = run_name(spec)
    set_seed(args.seed)
    net = build_net(cfg, args.mode, args.seed)
    frames = eval_frames(cfg, "train", stride=1)
    if args.limit_train:
        frames = frames[:: max(1, len(frames) // args.limit_train)][: args.limit_train]
    ds = TrainFrames(cfg, frames, augment=True)
    val_frames = eval_frames(cfg, "val")
    if args.limit_val:
        val_frames = val_frames[: args.limit_val]
    validator = Validator(cfg, val_frames)
    steps = len(ds) // tcfg["batch"]
    meta = {
        "base_model": tcfg["base_model"], "lora_rank": cfg["lora"]["rank"], "lora_regex": cfg["lora"]["target_regex"],
        "git_hash": git_hash(ignore=("results/phase3", "runs")),  # the queue's own logs don't count
        "teacher": tcfg["teacher"],
        "fingerprint": {
            "config_sha256": hashlib.sha256(Path("config.yaml").read_bytes()).hexdigest(),
            "code_sha256": code_sha256(), "n_train": len(ds), "n_val": len(validator.items),
            "steps_per_epoch": steps,
        },
    }
    out_root = Path(args.out_root or Path(cfg["paths"]["results_root"]) / "phase3" / "runs")
    ckpt_root = Path(args.ckpt_root or cfg["paths"]["checkpoint_root"])
    result = fit(
        net, forward_da3, lambda epoch, gen: loader(ds, tcfg, gen), lambda n: validator(n, DEVICE), spec, tcfg,
        out_dir=out_root / spec["name"], ckpt_dir=ckpt_root / spec["name"], meta=meta, device=DEVICE,
        steps_per_epoch=steps, max_epochs=args.epochs, tb_dir=Path(args.tb_root or "runs") / spec["name"],
        log=lambda msg: print(msg, flush=True),
    )
    print(json.dumps({"name": spec["name"], **result}), flush=True)
    return result


# --- overfit sanity check --------------------------------------------------------------------

def _colour(depth, lo=1.0, hi=40.0):
    t = np.clip((np.log(depth) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
    return cv2.applyColorMap((255 - t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def overlay(frame, pred, gt, masks, ecfg, path: Path, process_res: int) -> None:
    """RGB | aligned prediction, and the same prediction with LiDAR dots | per-point relative error."""
    rgb = load_resized(str(frame.image), process_res)[..., ::-1].copy()
    aligned, _ = align(pred, "depth", gt, masks["valid"], "median", ecfg["min_depth"], ecfg["max_depth"])
    pred_img = _colour(aligned)
    dots = pred_img.copy()
    err_img = (rgb * 0.4).astype(np.uint8)
    ys, xs = np.nonzero(masks["valid"])
    lidar = _colour(gt[ys, xs][None])[0]
    err = np.clip(np.abs(aligned[ys, xs] - gt[ys, xs]) / gt[ys, xs] / 0.25, 0, 1)
    for y, x, c, e in zip(ys, xs, lidar, err):
        cv2.circle(dots, (int(x), int(y)), 1, tuple(int(v) for v in c), -1)
        cv2.circle(err_img, (int(x), int(y)), 1, (0, int(255 * (1 - e)), int(255 * e)), -1)
    grid = np.vstack([np.hstack([rgb, pred_img]), np.hstack([dots, err_img])])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)


def overfit(cfg: dict, args) -> None:
    """Fits a few train frames without augmentation or distillation. Pass: AbsRel on those frames
    below 0.03 by the last step (criterion fixed in DECISIONS.md before running)."""
    tcfg, ecfg = cfg["train"], cfg["eval"]
    set_seed(0)
    net = build_net(cfg, "lora", 0)
    frames = eval_frames(cfg, "train", stride=1)
    frames = frames[:: len(frames) // args.overfit][: args.overfit]
    gts, masks, kept = [], [], []
    for f in frames:
        with np.load(frame_path(cfg["paths"]["cache_root"], f.seq, f.frame)) as g:
            gt = dict(g)
        m = frame_masks(gt, ecfg)
        if m is not None:  # eval would skip it
            gts.append(gt["depth"])
            masks.append(m)
            kept.append(f)
    frames = kept
    ds = TrainFrames(cfg, frames, augment=False)
    optimizer = torch.optim.AdamW(param_groups(net, args.lr, tcfg["lr_head"]), weight_decay=tcfg["weight_decay"])
    out_dir = Path(cfg["paths"]["results_root"]) / "overlays" / "phase3_overfit"

    @torch.no_grad()
    def score(step):
        net.eval()
        rows = []
        for i in range(len(ds)):
            out = forward_da3(net, ds[i]["image"][None].to(DEVICE))
            pred = out["depth"][0].float().cpu().numpy()
            rows.append(frame_metrics(pred, out["conf"][0].float().cpu().numpy(), gts[i], masks[i], ecfg))
            if i == 0 and step in (0, args.steps):
                overlay(frames[0], pred, gts[0], masks[0], ecfg,
                        out_dir / f"{frames[0].seq}_{frames[0].frame:06d}_step{step}.png", cfg["input"]["process_res"])
        net.train()
        m = mean_metrics(rows)
        print(f"overfit step {step}: absrel_all {m['absrel_all']:.4f} conf_median {m['conf_median']:.2f}", flush=True)
        return m

    m = score(0)
    step = 0
    gen = torch.Generator().manual_seed(0)
    while step < args.steps:
        for batch in loader(ds, tcfg, gen, workers=2):
            out = forward_da3(net, batch["image"].to(DEVICE))
            loss, _ = lidar_loss(out["depth"], out["conf"], batch["gt"].to(DEVICE), batch["mask"].to(DEVICE),
                                 tcfg["lambda_c"], cfg["eval"]["min_valid"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], tcfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 or step == args.steps:
                m = score(step)
            if step == args.steps:
                break
    verdict = "PASS" if m["absrel_all"] < 0.03 else "FAIL"
    print(f"overfit {args.overfit} frames, {args.steps} steps, lr {args.lr:g}: absrel_all {m['absrel_all']:.4f} -> {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["head", "lora"], default="lora")
    parser.add_argument("--lr", type=float, default=1e-4, help="LoRA rate; the head always uses train.lr_head")
    parser.add_argument("--lambda-d", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="stop after this many epochs (the schedule is unchanged)")
    parser.add_argument("--limit-train", type=int, help="debug: use this many train frames, evenly spaced")
    parser.add_argument("--limit-val", type=int, help="debug: use the first this many val frames")
    parser.add_argument("--out-root", help="debug: log directory root instead of results/phase3/runs")
    parser.add_argument("--ckpt-root", help="debug: checkpoint root instead of paths.checkpoint_root")
    parser.add_argument("--tb-root", help="debug: TensorBoard root instead of runs/")
    parser.add_argument("--overfit", type=int, help="run the overfit check on this many train frames")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if bool(args.out_root) != bool(args.ckpt_root):
        parser.error("give --out-root and --ckpt-root together, so debug runs never mix with real ones")
    if (args.limit_train or args.limit_val or args.epochs) and not args.out_root:
        parser.error("--limit-train, --limit-val and --epochs are for checks: also give --out-root and --ckpt-root")
    cfg = load_config()
    if args.overfit:
        overfit(cfg, args)
    else:
        train_run(cfg, args)


if __name__ == "__main__":
    main()
