import json
import math

import pytest
import torch

from src.train import fit, lr_factor, param_groups, resolve_stage, run_name

TCFG = {
    "batch": 2, "accumulate": 2, "epochs": 6, "patience": 2, "warmup_steps": 3, "weight_decay": 0.01,
    "grad_clip": 1.0, "lr_head": 5e-5, "lambda_c": 1.0, "distill_space": "linear",
}


# --- naming, schedule and parameter groups ---------------------------------------------------

def test_run_names():
    assert run_name({"mode": "lora", "lr": 1e-4, "lambda_d": 0.5, "seed": 0}) == "lora_lr1e-04_ld0.5_s0"
    assert run_name({"mode": "lora", "lr": 5e-5, "lambda_d": 0.0, "seed": 2}) == "lora_lr5e-05_ld0_s2"
    assert run_name({"mode": "head", "lambda_d": 0.1, "seed": 0}) == "head_ld0.1_s0"


def test_lr_factor_warms_up_then_decays_to_zero():
    f = [lr_factor(s, 10, 100) for s in range(100)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0)
    assert all(a <= b for a, b in zip(f[:10], f[1:10]))
    assert all(a >= b for a, b in zip(f[10:], f[11:]))
    assert f[55] == pytest.approx(0.5, abs=0.02)
    assert lr_factor(100, 10, 100) == pytest.approx(0.0) and lr_factor(150, 10, 100) == pytest.approx(0.0)


class Named(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.ModuleDict({"lora_A": torch.nn.Linear(2, 2), "frozen": torch.nn.Linear(2, 2)})
        self.head = torch.nn.Linear(2, 2)
        self.backbone["frozen"].requires_grad_(False)


def test_param_groups_give_lora_its_rate_and_the_head_lr_head():
    net = Named()
    groups = param_groups(net, lr=1e-4, lr_head=5e-5)
    by_lr = {g["lr"]: {id(p) for p in g["params"]} for g in groups}
    assert by_lr[1e-4] == {id(p) for p in net.backbone["lora_A"].parameters()}
    assert by_lr[5e-5] == {id(p) for p in net.head.parameters()}
    assert sum(len(g["params"]) for g in groups) == 4  # the frozen layer is left out


# --- stage resolution ------------------------------------------------------------------------

STAGES = [
    {"name": "S1", "select": True, "runs": [{"mode": "lora", "lr": 5e-5, "lambda_d": 0.5},
                                            {"mode": "lora", "lr": 1e-4, "lambda_d": 0.5}]},
    {"name": "S2", "select": True, "runs": [{"mode": "lora", "lr": "best", "lambda_d": 0.1},
                                            {"mode": "lora", "lr": "best", "lambda_d": 0.5}]},
    {"name": "S3", "select": False, "runs": [{"mode": "head", "lambda_d": "best"},
                                             {"mode": "lora", "lr": "best", "lambda_d": "best", "seed": 1}]},
]


def test_resolve_stage_fills_best_and_reuses_runs():
    s1 = resolve_stage(STAGES[0], None)
    assert [r["name"] for r in s1] == ["lora_lr5e-05_ld0.5_s0", "lora_lr1e-04_ld0.5_s0"]
    best = s1[1]
    s2 = resolve_stage(STAGES[1], best)
    assert [r["name"] for r in s2] == ["lora_lr1e-04_ld0.1_s0", "lora_lr1e-04_ld0.5_s0"]  # second = S1's
    s3 = resolve_stage(STAGES[2], s2[0])
    assert [r["name"] for r in s3] == ["head_ld0.1_s0", "lora_lr1e-04_ld0.1_s1"]
    assert "lr" not in s3[0]


def test_best_before_any_selection_is_an_error():
    with pytest.raises(ValueError, match="best"):
        resolve_stage(STAGES[1], None)


# --- the loop, on a tiny CPU net -------------------------------------------------------------

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.backbone = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.head = torch.nn.Conv2d(4, 2, 1)

    def forward(self, x):
        y = self.head(torch.relu(self.backbone(x)))
        return {"depth": torch.exp(y[:, 0]), "conf": torch.exp(y[:, 1]) + 1}


def tiny_forward(net, x):
    return net(x)


def make_loader(epoch, generator, n=6):
    """Fixed data; the order follows the per-epoch generator like a shuffled DataLoader."""
    g = torch.Generator().manual_seed(123)
    images = torch.rand(n, 3, 5, 7, generator=g)
    gt = torch.rand(n, 5, 7, generator=g) * 10 + 1
    teacher = torch.rand(n, 5, 7, generator=g) + 0.5
    order = torch.randperm(n, generator=generator)
    for i in range(0, n, 2):
        idx = order[i : i + 2]
        yield {"image": images[idx], "gt": gt[idx], "mask": gt[idx] < 9, "teacher": teacher[idx]}


def scripted_validate(values):
    calls = iter(values)
    return lambda net: {"absrel_all": next(calls), "absrel_grass": 0.0}


def run_fit(tmp_path, validate, max_epochs=None, name="r"):
    net = Tiny()
    spec = {"name": name, "mode": "lora", "lr": 1e-2, "lambda_d": 0.5, "seed": 0}
    return net, fit(
        net, tiny_forward, make_loader, validate, spec, TCFG, out_dir=tmp_path / "out" / name,
        ckpt_dir=tmp_path / "ckpt" / name, meta={"base_model": "tiny"}, device="cpu", steps_per_epoch=3,
        max_epochs=max_epochs, log=lambda *a: None,
    )


def test_early_stopping_and_best_checkpoint(tmp_path):
    # epoch 0 is the untrained model; then 1..: best at epoch 2, no gain for 2 epochs -> stop at 4
    values = [0.9, 0.5, 0.4, 0.45, 0.41, 0.3, 0.2]
    net, result = run_fit(tmp_path, scripted_validate(values))
    assert result["best_epoch"] == 2 and result["epochs_run"] == 4 and result["stopped"] == "early"
    best = torch.load(tmp_path / "ckpt/r/best_val.pt", weights_only=False)
    assert best["epoch"] == 2 and best["val"]["absrel_all"] == 0.4
    done = json.loads((tmp_path / "out/r/done.json").read_text())
    assert done["best_epoch"] == 2
    lines = (tmp_path / "out/r/log.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + 5  # header, epoch 0, epochs 1-4


def test_epoch_zero_does_not_count_as_a_candidate(tmp_path):
    values = [0.1, 0.5, 0.6, 0.7, 0.8]
    _, result = run_fit(tmp_path, scripted_validate(values))
    assert result["best_epoch"] == 1


def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    values = [0.9, 0.5, 0.4, 0.35, 0.3, 0.29, 0.28]
    full_net, full = run_fit(tmp_path, scripted_validate(values), name="full")
    _, partial = run_fit(tmp_path, scripted_validate(values[:3]), max_epochs=2, name="part")
    assert partial["stopped"] == "max_epochs"
    resumed_net, resumed = run_fit(tmp_path, scripted_validate(values[3:]), name="part")
    assert resumed["epochs_run"] == full["epochs_run"] == 6
    for a, b in zip(full_net.parameters(), resumed_net.parameters()):
        assert torch.equal(a, b)
    full_log = (tmp_path / "out/full/log.csv").read_text().splitlines()
    part_log = (tmp_path / "out/part/log.csv").read_text().splitlines()
    strip = lambda rows: [",".join(r.split(",")[:-1]) for r in rows]  # drop wall-clock column
    assert strip(full_log) == strip(part_log)


def test_finished_run_is_not_rerun(tmp_path):
    values = [0.9, 0.5, 0.4, 0.45, 0.41]
    run_fit(tmp_path, scripted_validate(values))
    _, again = run_fit(tmp_path, scripted_validate([]))
    assert again["best_epoch"] == 2


# --- review fixes (2026-11-06) ---------------------------------------------------------------

def run_fit2(tmp_path, validate, name="r", forward=tiny_forward, meta=None, max_epochs=None):
    net = Tiny()
    spec = {"name": name, "mode": "lora", "lr": 1e-2, "lambda_d": 0.5, "seed": 0}
    return net, fit(net, forward, make_loader, validate, spec, TCFG, out_dir=tmp_path / "out" / name,
                    ckpt_dir=tmp_path / "ckpt" / name, meta=meta or {"fingerprint": {"cfg": "a"}},
                    device="cpu", steps_per_epoch=3, max_epochs=max_epochs, log=lambda *a: None)


def test_resume_refuses_a_different_fingerprint(tmp_path):
    run_fit2(tmp_path, scripted_validate([0.9, 0.5]), max_epochs=1)
    with pytest.raises(ValueError, match="refusing to resume"):
        run_fit2(tmp_path, scripted_validate([0.4]), meta={"fingerprint": {"cfg": "b"}})


def test_non_finite_loss_finishes_the_run_with_its_best_so_far(tmp_path):
    calls = {"n": 0}

    def flaky(net, x):
        calls["n"] += 1
        out = net(x)
        return {"depth": out["depth"] * (float("nan") if calls["n"] > 3 else 1.0), "conf": out["conf"]}

    _, result = run_fit2(tmp_path, scripted_validate([0.9, 0.5, 0.4]), forward=flaky)
    assert result["stopped"] == "nonfinite" and result["best_epoch"] == 1
    assert json.loads((tmp_path / "out/r/done.json").read_text())["stopped"] == "nonfinite"


def test_non_finite_from_the_start_is_ranked_last(tmp_path):
    from src.train import select_best

    def bad(net, x):
        out = net(x)
        return {"depth": out["depth"] * float("nan"), "conf": out["conf"]}

    _, result = run_fit2(tmp_path, scripted_validate([0.9]), forward=bad)
    assert result["best_epoch"] is None and result["best_val"]["val_absrel_all"] == math.inf
    specs = [{"name": "a"}, {"name": "b"}]
    ok = {"best_val": {"val_absrel_all": 0.5}}
    assert select_best(specs, {"a": result, "b": ok})["name"] == "b"
    nan = {"best_val": {"val_absrel_all": float("nan")}}
    assert select_best(specs, {"a": nan, "b": ok})["name"] == "b"


def test_non_finite_val_stops_the_run(tmp_path):
    _, result = run_fit2(tmp_path, scripted_validate([0.9, 0.5, float("nan"), 0.1]))
    assert result["stopped"] == "nonfinite" and result["best_epoch"] == 1 and result["epochs_run"] == 1
    assert "epoch 2" in result["nonfinite"]  # the uncommitted epoch is named, not logged


def test_crash_before_last_pt_discards_the_pending_best(tmp_path, monkeypatch):
    """Epoch 2 looked best, then the run died before last.pt. The replay of epoch 2 is worse, so
    best_val.pt must still hold epoch 1."""
    import src.train as train

    real_save = train._save

    def crash_on_second_last(obj, path):
        if path.name == "last.pt" and obj["progress"]["epoch"] == 2:
            raise RuntimeError("power cut")
        real_save(obj, path)

    monkeypatch.setattr(train, "_save", crash_on_second_last)
    with pytest.raises(RuntimeError):
        run_fit2(tmp_path, scripted_validate([0.9, 0.5, 0.1]))
    monkeypatch.setattr(train, "_save", real_save)
    _, result = run_fit2(tmp_path, scripted_validate([0.6, 0.7, 0.8]))
    assert result["best_epoch"] == 1
    assert torch.load(tmp_path / "ckpt/r/best_val.pt", weights_only=False)["epoch"] == 1
    assert not list((tmp_path / "ckpt/r").glob("*pending*"))


def test_crash_after_last_pt_keeps_the_best_and_the_log(tmp_path, monkeypatch):
    import src.train as train

    real_log = train._write_log

    def crash_on_epoch_two(rows, path):
        if rows[-1]["epoch"] == 2:
            raise RuntimeError("power cut")
        real_log(rows, path)

    monkeypatch.setattr(train, "_write_log", crash_on_epoch_two)
    with pytest.raises(RuntimeError):
        run_fit2(tmp_path, scripted_validate([0.9, 0.5, 0.1]), max_epochs=2)
    monkeypatch.setattr(train, "_write_log", real_log)
    _, result = run_fit2(tmp_path, scripted_validate([]), max_epochs=2)  # nothing left to train
    assert result["best_epoch"] == 2
    assert torch.load(tmp_path / "ckpt/r/best_val.pt", weights_only=False)["epoch"] == 2
    lines = (tmp_path / "out/r/log.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + 3


def test_sparse_images_are_skipped_like_eval():
    from src.losses import lidar_loss

    depth = torch.rand(2, 4, 5) + 1
    gt = torch.rand(2, 4, 5) * 5 + 1
    conf = torch.ones(2, 4, 5) * 2
    mask = torch.ones(2, 4, 5, dtype=torch.bool)
    mask[1] = False
    mask[1, 0, :3] = True  # 3 pixels: under min_valid
    both, _ = lidar_loss(depth, conf, gt, mask, 1.0, min_valid=5)
    first, _ = lidar_loss(depth[:1], conf[:1], gt[:1], mask[:1], 1.0, min_valid=5)
    assert torch.allclose(both, first)


def test_run_name_refuses_ambiguous_rates():
    with pytest.raises(ValueError, match="name"):
        run_name({"mode": "lora", "lr": 1.5e-4, "lambda_d": 0.5, "seed": 0})
