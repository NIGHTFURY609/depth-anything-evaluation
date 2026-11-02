import csv

import pytest
import torch

import phase0_check
from src.config import load_config


def raise_oom(*args, **kwargs):
    raise torch.cuda.OutOfMemoryError("injected")


def test_oom_recorded(monkeypatch):
    monkeypatch.setattr(phase0_check, "load_da3", raise_oom)
    row = phase0_check.measure_train("any/model", "lora", 4, False, load_config())
    assert row["status"] == "OOM"
    assert row["kind"] == "train"
    assert (row["mode"], row["batch"], row["checkpointing"]) == ("lora", 4, False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_free_vram_recorded_even_on_oom(monkeypatch):
    monkeypatch.setattr(phase0_check, "load_da3", raise_oom)
    row = phase0_check.measure_train("any/model", "head", 2, False, load_config())
    assert row["free_before_mb"] > 0


def test_oom_retried_with_checkpointing(monkeypatch):
    calls = []

    def fake_measure(model_id, mode, batch, checkpointing, cfg):
        calls.append(checkpointing)
        return {"status": "OOM" if not checkpointing else "ok", "checkpointing": checkpointing}

    monkeypatch.setattr(phase0_check, "measure_train", fake_measure)
    rows = phase0_check.measure_train_with_retry("any/model", "full", 4, load_config())
    assert calls == [False, True]
    assert [r["status"] for r in rows] == ["OOM", "ok"]


def test_no_retry_when_step_fits(monkeypatch):
    monkeypatch.setattr(
        phase0_check,
        "measure_train",
        lambda model_id, mode, batch, checkpointing, cfg: {"status": "ok"},
    )
    assert len(phase0_check.measure_train_with_retry("any/model", "head", 2, load_config())) == 1


@pytest.mark.gpu
def test_da2_runs_in_full_precision():
    """Baselines must match the official float32 pipeline, not a bf16-quantised one."""
    cfg = load_config()
    run = phase0_check.load_da2_runner(cfg["models"]["da2_small"])
    with torch.no_grad():
        depth, conf = run(phase0_check._sample_batch(1, cfg))
    assert depth.dtype == torch.float32
    assert conf is None


def test_csv_columns(tmp_path):
    path = tmp_path / "phase0.csv"
    phase0_check.write_rows([{"model": "m", "status": "ok"}], path)
    with open(path) as f:
        header = next(csv.reader(f))
    assert header == [
        "model", "kind", "mode", "batch", "checkpointing", "has_conf", "out_h", "out_w",
        "peak_alloc_mb", "peak_reserved_mb", "free_before_mb", "latency_ms_median", "status",
    ]
