from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.gt import frame_path
from src.models import preprocess
from src.rellis import Frame
from src.train_data import TrainFrames, load_resized, normalize

H, W = 322, 504


def camera_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (12, 20, 3), dtype=np.uint8)
    return np.kron(coarse, np.ones((100, 96, 1), dtype=np.uint8)) + rng.integers(0, 3, (1200, 1920, 3), dtype=np.uint8)


def fake_cfg(root: Path) -> dict:
    return {
        "paths": {"cache_root": str(root / "cache"), "data_root": str(root)},
        "input": {"process_res": 504},
        "split": {"train": ["00000"], "val": "00001", "test": "00002"},
        "eval": {"min_depth": 1.0, "max_depth": 80.0},
        "train": {"teacher": "pred/da3_small", "flip_p": 0.0,
                  "jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02}},
    }


def fake_frame(root: Path, seq: str = "00000", number: int = 7) -> Frame:
    image = root / f"frame{number:06d}.jpg"
    cv2.imwrite(str(image), camera_image(number)[..., ::-1])
    rng = np.random.default_rng(number)
    depth = np.where(rng.random((H, W)) < 0.1, rng.uniform(0.5, 90, (H, W)), 0).astype(np.float32)
    gt_path = frame_path(root / "cache", seq, number)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(gt_path, depth=depth, keep=depth > 0, labels=np.zeros((H, W), np.uint8))
    teacher = frame_path(root / "cache", seq, number, "pred/da3_small")
    teacher.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(teacher, pred=rng.uniform(1, 5, (H, W)).astype(np.float32))
    return Frame(seq, number, image, Path("s"), Path("l"), None)


def test_load_resized_matches_preprocess():
    image = camera_image(0)
    x = normalize(torch.from_numpy(load_resized(image, 504)).permute(2, 0, 1).float() / 255)
    assert torch.equal(x, preprocess([image])[0])


def test_unaugmented_item_matches_preprocess_and_caches(tmp_path):
    frame = fake_frame(tmp_path)
    ds = TrainFrames(fake_cfg(tmp_path), [frame], augment=False)
    item = ds[0]
    assert torch.equal(item["image"], preprocess([str(frame.image)])[0])
    with np.load(frame_path(tmp_path / "cache", "00000", 7)) as g:
        depth = g["depth"]
    valid = (depth >= 1) & (depth <= 80)
    assert torch.equal(item["mask"], torch.from_numpy(valid))
    assert torch.equal(item["gt"], torch.from_numpy(depth))
    assert item["teacher"].shape == (H, W)


def test_flip_mirrors_image_gt_and_teacher_together(tmp_path):
    frame = fake_frame(tmp_path)
    cfg = fake_cfg(tmp_path)
    plain = TrainFrames(cfg, [frame], augment=False)[0]
    cfg["train"]["flip_p"] = 1.0
    cfg["train"]["jitter"] = {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0}
    flipped = TrainFrames(cfg, [frame], augment=True)[0]
    for key in ("image", "gt", "mask", "teacher"):
        assert torch.equal(flipped[key], plain[key].flip(-1)), key


def test_jitter_changes_image_only(tmp_path):
    frame = fake_frame(tmp_path)
    cfg = fake_cfg(tmp_path)
    plain = TrainFrames(cfg, [frame], augment=False)[0]
    torch.manual_seed(1)
    jittered = TrainFrames(cfg, [frame], augment=True)[0]
    assert not torch.equal(jittered["image"], plain["image"])
    for key in ("gt", "mask", "teacher"):
        assert torch.equal(jittered[key], plain[key]), key


@pytest.mark.parametrize("seq", ["00001", "00002"])
def test_refuses_val_and_test_frames(tmp_path, seq):
    frame = Frame(seq, 0, Path("x.jpg"), Path("s"), Path("l"), None)
    with pytest.raises(ValueError, match="not a train sequence"):
        TrainFrames(fake_cfg(tmp_path), [frame])


def test_loader_with_workers_repeats_for_the_same_seed_and_epoch(tmp_path):
    """Resume relies on (seed, epoch) fixing the shuffle, the flips and the jitter in workers."""
    from torch.utils.data import DataLoader

    frames = [fake_frame(tmp_path, number=n) for n in range(6)]
    cfg = fake_cfg(tmp_path)
    cfg["train"]["flip_p"] = 0.5
    ds = TrainFrames(cfg, frames, augment=True)

    def epoch(seed):
        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2,
                            generator=torch.Generator().manual_seed(seed))
        return [(b["index"].tolist(), b["image"].sum().item(), b["gt"].sum().item()) for b in loader]

    assert epoch(1001) == epoch(1001)
    assert epoch(1001) != epoch(1002)
