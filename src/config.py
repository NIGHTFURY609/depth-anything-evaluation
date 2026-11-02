import os
import random

import numpy as np
import torch
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: os.path.expanduser(v) for k, v in cfg["paths"].items()}
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
