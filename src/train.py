"""Phase 3 training loop: confidence-weighted LiDAR loss plus distillation, early stopping on val.

`fit` is model-agnostic (it takes the net, a forward function, a loader factory and a validator),
so tests/test_train.py runs it on a tiny CPU net. Each epoch's shuffle and augmentation are seeded
from (seed, epoch), and last.pt holds the optimiser and schedule, so a resumed run repeats an
uninterrupted one.
"""

import csv
import json
import math
import os
import time
from pathlib import Path

import torch

from src.losses import distill_loss, lidar_loss
from src.models import load_trainable, trainable_state

TRAIN_KEYS = ["epoch", "step", "lr", "lr_head", "loss", "lidar", "distill", "train_absrel"]
TAIL_KEYS = ["best", "time_s"]


def run_name(spec: dict) -> str:
    seed, ld = spec.get("seed", 0), f"{spec['lambda_d']:g}"
    if spec["mode"] == "head":
        return f"head_ld{ld}_s{seed}"
    return f"{spec['mode']}_lr{spec['lr']:.0e}_ld{ld}_s{seed}"


def resolve_stage(stage: dict, best: dict | None) -> list[dict]:
    """Concrete run specs for a stage. "best" takes that field from the best run selected so far."""
    runs = []
    for entry in stage["runs"]:
        spec = {"seed": 0, **entry}
        for key, value in spec.items():
            if value == "best":
                if best is None:
                    raise ValueError(f"stage {stage['name']} uses 'best' before any run was selected")
                spec[key] = best[key]
        if spec["mode"] == "head":
            spec.pop("lr", None)
        spec["name"] = run_name(spec)
        runs.append(spec)
    return runs


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup to 1 over `warmup` optimiser steps, then cosine to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def param_groups(net: torch.nn.Module, lr: float, lr_head: float) -> list[dict]:
    """The head at lr_head in every mode; LoRA (and any other trainable backbone) parameters at lr."""
    head, rest = [], []
    for name, p in net.named_parameters():
        if p.requires_grad:
            (head if name.startswith("head.") else rest).append(p)
    return [g for g in ({"params": rest, "lr": lr}, {"params": head, "lr": lr_head}) if g["params"]]


def _save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_log(rows: list[dict], path: Path) -> None:
    val_keys = [k for r in rows for k in r if k not in TRAIN_KEYS + TAIL_KEYS]
    fields = TRAIN_KEYS + list(dict.fromkeys(val_keys)) + TAIL_KEYS
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _done(out_dir: Path, progress: dict, stopped: str, spec: dict) -> dict:
    best = progress["rows"][progress["best_epoch"]]
    result = {
        "spec": spec,
        "best_epoch": progress["best_epoch"], "epochs_run": progress["epoch"], "stopped": stopped,
        "train_seconds": round(progress["seconds"]), "best_val": {k: v for k, v in best.items() if k.startswith("val_")},
        "epoch0_val": {k: v for k, v in progress["rows"][0].items() if k.startswith("val_")},
    }
    if stopped != "max_epochs":
        (out_dir / "done.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def fit(net, forward, make_loader, validate, spec: dict, tcfg: dict, out_dir, ckpt_dir, meta: dict,
        device: str, steps_per_epoch: int, max_epochs: int | None = None, log=print, tb_dir=None) -> dict:
    """Trains until early stopping or the end of the schedule; resumes from ckpt_dir/last.pt.

    make_loader(epoch, generator) yields batches with image, gt, mask and teacher. validate(net)
    returns val metrics including `absrel_all`, the selection metric. Epoch 0 is the untrained
    net, logged for reference but never a candidate. `max_epochs` stops early without finishing
    the run (for sanity checks); the schedule always spans tcfg["epochs"].
    """
    out_dir, ckpt_dir = Path(out_dir), Path(ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "done.json").exists():
        return json.loads((out_dir / "done.json").read_text())

    accumulate = tcfg["accumulate"]
    groups = param_groups(net, spec.get("lr", tcfg["lr_head"]), tcfg["lr_head"])
    trainable = [p for g in groups for p in g["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=tcfg["weight_decay"])
    total = tcfg["epochs"] * math.ceil(steps_per_epoch / accumulate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, tcfg["warmup_steps"], total))
    meta = {**meta, **{k: v for k, v in spec.items()}}
    writer = None
    if tb_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tb_dir)

    last = ckpt_dir / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        if ckpt["meta"].get("name") != spec["name"]:
            raise ValueError(f"{last} belongs to {ckpt['meta'].get('name')}, not {spec['name']}")
        load_trainable(net, ckpt["state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        progress = ckpt["progress"]
        log(f"{spec['name']}: resuming after epoch {progress['epoch']}")
    else:
        start = time.time()
        val = validate(net)
        progress = {"epoch": 0, "step": 0, "best": math.inf, "best_epoch": None, "seconds": 0.0, "rows": [
            {"epoch": 0, "step": 0, **{f"val_{k}": v for k, v in val.items()}, "best": False,
             "time_s": round(time.time() - start, 1)}]}
        _write_log(progress["rows"], out_dir / "log.csv")
        log(f"{spec['name']}: epoch 0 (untrained) val absrel_all {val['absrel_all']:.4f}")

    max_epochs = tcfg["epochs"] if max_epochs is None else min(max_epochs, tcfg["epochs"])
    lambda_d, lambda_c, space = spec["lambda_d"], tcfg["lambda_c"], tcfg["distill_space"]
    while True:
        if progress["best_epoch"] is not None and progress["epoch"] - progress["best_epoch"] >= tcfg["patience"]:
            return _done(out_dir, progress, "early", spec)
        if progress["epoch"] >= tcfg["epochs"]:
            return _done(out_dir, progress, "schedule_end", spec)
        if progress["epoch"] >= max_epochs:
            return _done(out_dir, progress, "max_epochs", spec)

        epoch = progress["epoch"] + 1
        seed = spec["seed"] * 1000 + epoch
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        start = time.time()
        net.train()
        sums = {"loss": 0.0, "lidar": 0.0, "distill": 0.0, "train_absrel": 0.0}
        n_batches = 0
        for i, batch in enumerate(make_loader(epoch, generator)):
            x = batch["image"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            out = forward(net, x)
            l_lidar, stats = lidar_loss(out["depth"], out["conf"], gt, mask, lambda_c)
            if lambda_d > 0:
                l_distill = distill_loss(out["depth"], batch["teacher"].to(device, non_blocking=True), space)
            else:
                l_distill = torch.zeros((), device=l_lidar.device)
            loss = l_lidar + lambda_d * l_distill
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec['name']}: non-finite loss at epoch {epoch}, batch {i}")
            (loss / accumulate).backward()
            if (i + 1) % accumulate == 0 or i + 1 == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(trainable, tcfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                progress["step"] += 1
            sums["loss"] += loss.item()
            sums["lidar"] += l_lidar.item()
            sums["distill"] += l_distill.item()
            sums["train_absrel"] += 0.0 if math.isnan(stats["absrel"]) else stats["absrel"]
            n_batches += 1

        val = validate(net)
        lrs = [g["lr"] for g in optimizer.param_groups]
        row = {"epoch": epoch, "step": progress["step"], "lr": lrs[0], "lr_head": lrs[-1],
               **{k: v / max(n_batches, 1) for k, v in sums.items()},
               **{f"val_{k}": v for k, v in val.items()}}
        improved = val["absrel_all"] < progress["best"]
        if improved:
            progress["best"], progress["best_epoch"] = val["absrel_all"], epoch
            _save({"state": trainable_state(net), "meta": {**meta, "epoch": epoch, "val": val},
                   "epoch": epoch, "val": val}, ckpt_dir / "best_val.pt")
        progress["seconds"] += time.time() - start
        row["best"] = improved
        row["time_s"] = round(time.time() - start, 1)
        progress["rows"].append(row)
        progress["epoch"] = epoch
        _save({"state": trainable_state(net), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "progress": progress, "meta": meta}, last)
        _write_log(progress["rows"], out_dir / "log.csv")
        if writer is not None:
            for k, v in row.items():
                if isinstance(v, (int, float)) and k != "epoch":
                    writer.add_scalar(k, v, epoch)
            writer.flush()
        log(f"{spec['name']}: epoch {epoch}/{tcfg['epochs']} loss {row['loss']:.4f} "
            f"val absrel_all {val['absrel_all']:.4f} (best {progress['best']:.4f} @ {progress['best_epoch']}) "
            f"{row['time_s']:.0f}s")


def select_best(specs: list[dict], results: dict[str, dict | None], metric: str = "absrel_all") -> dict:
    """The stage's run with the lowest best-epoch val metric; earlier runs win ties.

    results maps run name to its done.json (None if unfinished). Every run must be finished.
    """
    missing = [s["name"] for s in specs if results.get(s["name"]) is None]
    if missing:
        raise ValueError(f"cannot select: unfinished runs {missing}")
    return min(specs, key=lambda s: results[s["name"]]["best_val"][f"val_{metric}"])
