from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_config
from src.infer import eval_frames, load_runner, write_predictions
from src.rellis import Frame


def fake_frames(seq: str, n: int) -> list[Frame]:
    return [Frame(seq, i, Path(f"{i}.jpg"), Path("s"), Path("l"), None) for i in range(n)]


def test_eval_frames_takes_every_stride_th_frame(monkeypatch):
    monkeypatch.setattr("src.infer.list_frames", lambda root, seq: fake_frames(seq, 12))
    cfg = {"paths": {"data_root": "x"}, "split": {"val": "00001", "eval_stride": 5},
           "gt": {"dropped_frames": {}}}
    assert [f.frame for f in eval_frames(cfg, "val")] == [0, 5, 10]


def test_eval_frames_skips_dropped_frames(monkeypatch):
    monkeypatch.setattr("src.infer.list_frames", lambda root, seq: fake_frames(seq, 12))
    cfg = {"paths": {"data_root": "x"}, "split": {"test": "00002", "eval_stride": 5},
           "gt": {"dropped_frames": {"00002": [5]}}}
    assert [f.frame for f in eval_frames(cfg, "test")] == [0, 10]


def test_eval_frames_covers_every_train_sequence(monkeypatch):
    monkeypatch.setattr("src.infer.list_frames", lambda root, seq: fake_frames(seq, 3))
    cfg = {"paths": {"data_root": "x"}, "split": {"train": ["00000", "00003"], "eval_stride": 5},
           "gt": {"dropped_frames": {}}}
    frames = eval_frames(cfg, "train", stride=1)
    assert [(f.seq, f.frame) for f in frames] == [
        ("00000", 0), ("00000", 1), ("00000", 2), ("00003", 0), ("00003", 1), ("00003", 2),
    ]


def test_write_predictions_stores_pred_conf_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infer.preprocess", lambda paths, res: torch.zeros(1, 3, 322, 504))

    def run(x):
        return torch.full((1, 322, 504), 2.0), torch.full((1, 322, 504), 3.0)

    frames = fake_frames("00001", 2)
    paths = write_predictions(run, frames, tmp_path, "da3_small", "repo/id", "depth", "abc123", 504)
    assert len(paths) == 2
    with np.load(paths[1]) as p:
        assert p["pred"].dtype == np.float32 and p["pred"].shape == (322, 504)
        assert np.all(p["conf"] == 3.0)
        assert (str(p["model_id"]), str(p["kind"]), str(p["git_hash"])) == ("repo/id", "depth", "abc123")
    assert paths[1] == tmp_path / "pred" / "da3_small" / "00001" / "000001.npz"


def test_write_predictions_omits_conf_when_model_has_none(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infer.preprocess", lambda paths, res: torch.zeros(1, 3, 322, 504))
    paths = write_predictions(
        lambda x: (torch.ones(1, 322, 504), None), fake_frames("00001", 1), tmp_path,
        "da2_small", "repo/id", "disparity", "abc123", 504,
    )
    with np.load(paths[0]) as p:
        assert "conf" not in p.files


@pytest.mark.gpu
def test_cached_da3_matches_live_forward(tmp_path):
    from src.models import forward_da3, load_da3, preprocess

    cfg = load_config()
    frame = eval_frames(cfg, "val")[0]
    run, kind = load_runner(cfg, "da3_small", "cuda")
    assert kind == "depth"
    [path] = write_predictions(run, [frame], tmp_path, "da3_small", "id", kind, "h", 504)
    net = load_da3(cfg["models"]["da3_small"], "cuda").model
    with torch.no_grad():
        live = forward_da3(net, preprocess([str(frame.image)]).to("cuda"))
    with np.load(path) as p:
        np.testing.assert_allclose(p["pred"], live["depth"][0].float().cpu().numpy(), rtol=1e-5)
        np.testing.assert_allclose(p["conf"], live["conf"][0].float().cpu().numpy(), rtol=1e-5)
