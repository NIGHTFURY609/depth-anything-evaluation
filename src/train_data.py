"""Training frames: image, LiDAR ground truth and cached teacher depth, flipped together.

Images go through the same steps as upstream's InputProcessor (src.models.preprocess), with
colour jitter inserted before normalisation. Only train sequences are accepted.
"""

import numpy as np
import torch
from depth_anything_3.utils.io.input_processor import InputProcessor
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from src.eval import valid_mask
from src.gt import frame_path

_PROC = InputProcessor()


def load_resized(image, process_res: int) -> np.ndarray:
    """(H, W, 3) uint8 at the model's input size, exactly as upstream resizes for inference."""
    pil = _PROC._load_image(image)
    pil = _PROC._resize_image(pil, process_res, "upper_bound_resize")
    pil = _PROC._make_divisible_by_resize(pil, _PROC.PATCH_SIZE)
    return np.asarray(pil)


def to_unit(image: np.ndarray) -> torch.Tensor:
    """(3, H, W) float in [0, 1], as torchvision's ToTensor gives."""
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255


def normalize(x: torch.Tensor) -> torch.Tensor:
    return _PROC.NORMALIZE(x)


class TrainFrames(Dataset):
    def __init__(self, cfg: dict, frames: list, augment: bool = True):
        train_seqs = set(cfg["split"]["train"])
        bad = sorted({f.seq for f in frames} - train_seqs)
        if bad:
            raise ValueError(f"{bad} is not a train sequence")
        self.frames = frames
        self.cache = cfg["paths"]["cache_root"]
        self.process_res = cfg["input"]["process_res"]
        self.dmin, self.dmax = cfg["eval"]["min_depth"], cfg["eval"]["max_depth"]
        tcfg = cfg["train"]
        self.teacher = tcfg["teacher"]
        self.augment = augment
        self.flip_p = tcfg["flip_p"]
        self.jitter = ColorJitter(**tcfg["jitter"])

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> dict:
        f = self.frames[i]
        image = to_unit(load_resized(str(f.image), self.process_res))
        with np.load(frame_path(self.cache, f.seq, f.frame)) as g:
            depth, keep = g["depth"], g["keep"]
        with np.load(frame_path(self.cache, f.seq, f.frame, self.teacher)) as t:
            teacher = t["pred"].astype(np.float32)
        item = {
            "gt": torch.from_numpy(depth.astype(np.float32)),
            "mask": torch.from_numpy(valid_mask(depth, keep, self.dmin, self.dmax)),
            "teacher": torch.from_numpy(teacher),
        }
        if self.augment:
            image = self.jitter(image)
        item["image"] = normalize(image)
        shapes = {k: tuple(v.shape[-2:]) for k, v in item.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"{f.seq}/{f.frame}: mismatched shapes {shapes}")
        if self.augment and torch.rand(()) < self.flip_p:
            item = {k: v.flip(-1) for k, v in item.items()}
        item["index"] = i
        return item
