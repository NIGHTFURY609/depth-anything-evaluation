"""Phase 0: per-model outputs, inference VRAM and latency, and DA3-Small training VRAM."""

import csv
import gc
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForDepthEstimation

from src.config import load_config, set_seed
from src.models import enable_checkpointing, forward_da3, load_da3, preprocess, set_trainable

COLUMNS = [
    "model", "kind", "mode", "batch", "checkpointing", "has_conf", "out_h", "out_w",
    "peak_alloc_mb", "peak_reserved_mb", "free_before_mb", "latency_ms_median", "status",
]
DEVICE = "cuda"
MB = 1024**2
TRAIN_STEPS = 4  # The first step allocates optimiser state; later ones show the true peak.


def _free_mb() -> float:
    return torch.cuda.mem_get_info()[0] / MB if torch.cuda.is_available() else 0.0


def _reset_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def _peaks() -> dict:
    return {
        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / MB),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / MB),
    }


def _sample_batch(batch: int, cfg: dict) -> torch.Tensor:
    """A seeded camera-sized image through the real preprocessing."""
    rng = np.random.default_rng(cfg["seed"])
    image = rng.integers(0, 255, (1200, 1920, 3), dtype=np.uint8)
    return preprocess([image] * batch, cfg["input"]["process_res"]).to(DEVICE)


def load_da3_runner(model_id: str):
    net = load_da3(model_id, DEVICE).model

    def run(x):
        out = forward_da3(net, x)
        return out["depth"], out["conf"]

    return run


def load_da2_runner(model_id: str):
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(DEVICE).eval()

    # float32, like the official pipeline: bf16 would quantise the disparity baselines.
    def run(x):
        return model(pixel_values=x).predicted_depth, None

    return run


@torch.no_grad()
def measure_inference(load_fn, model_id: str, cfg: dict) -> dict:
    row = {"model": model_id, "kind": "inference", "mode": "-", "batch": 1, "checkpointing": False}
    _reset_gpu()
    row["free_before_mb"] = round(_free_mb())
    run = None
    try:
        run = load_fn(model_id)
        x = _sample_batch(1, cfg)
        for _ in range(cfg["phase0"]["latency_warmup"]):
            run(x)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(cfg["phase0"]["latency_runs"]):
            (depth, conf), ms = _timed(lambda: run(x))
            times.append(ms)
        row.update(_peaks())
        row.update(
            has_conf=conf is not None,
            out_h=depth.shape[-2],
            out_w=depth.shape[-1],
            latency_ms_median=round(statistics.median(times), 1),
            status="ok",
        )
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    finally:
        del run
        _reset_gpu()
    return row


def measure_train(model_id: str, mode: str, batch: int, checkpointing: bool, cfg: dict) -> dict:
    row = {
        "model": model_id, "kind": "train", "mode": mode, "batch": batch,
        "checkpointing": checkpointing,
    }
    _reset_gpu()
    row["free_before_mb"] = round(_free_mb())
    net = optimizer = None
    try:
        net = load_da3(model_id, DEVICE).model
        set_trainable(net, mode, cfg["lora"]["rank"], cfg["lora"]["target_regex"])
        if checkpointing:
            enable_checkpointing(net)
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-5)
        x = _sample_batch(batch, cfg)

        def step():
            out = forward_da3(net, x)
            (out["depth"].mean() + out["conf"].mean()).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return out["depth"].shape

        times = []
        for _ in range(TRAIN_STEPS):
            shape, ms = _timed(step)
            times.append(ms)
        row.update(_peaks())
        row.update(
            has_conf=True,
            out_h=shape[-2],
            out_w=shape[-1],
            latency_ms_median=round(statistics.median(times[1:]), 1),
            status="ok",
        )
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    finally:
        del net, optimizer
        _reset_gpu()
    return row


def measure_train_with_retry(model_id: str, mode: str, batch: int, cfg: dict) -> list[dict]:
    rows = [measure_train(model_id, mode, batch, False, cfg)]
    if rows[0]["status"] == "OOM":
        rows.append(measure_train(model_id, mode, batch, True, cfg))
    return rows


def write_rows(rows: list[dict], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    models = cfg["models"]
    path = Path(cfg["paths"]["results_root"]) / "phase0.csv"

    rows = []
    for load_fn, key in [
        (load_da3_runner, "da3_small"),
        (load_da3_runner, "da3_base"),
        (load_da2_runner, "da2_small"),
        (load_da2_runner, "da2_base"),
    ]:
        rows.append(measure_inference(load_fn, models[key], cfg))
        print(rows[-1], flush=True)
        write_rows(rows, path)

    for mode in cfg["phase0"]["modes"]:
        for batch in cfg["phase0"]["batch_sizes"]:
            for row in measure_train_with_retry(models["da3_small"], mode, batch, cfg):
                rows.append(row)
                print(row, flush=True)
            write_rows(rows, path)


if __name__ == "__main__":
    main()
