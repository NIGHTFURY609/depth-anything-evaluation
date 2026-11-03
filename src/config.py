import os
import random
import subprocess

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


def git_hash() -> str:
    """HEAD commit, suffixed with -dirty if the tree has uncommitted changes."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return head.stdout.strip() + ("-dirty" if status.stdout.strip() else "")
