import os

import torch

from src.config import load_config, set_seed


def test_config_has_input_size():
    cfg = load_config()
    assert (cfg["input"]["width"], cfg["input"]["height"]) == (504, 322)
    assert cfg["input"]["width"] % 14 == 0
    assert cfg["input"]["height"] % 14 == 0


def test_config_expands_home_in_paths():
    cfg = load_config()
    assert cfg["paths"]["data_root"].startswith(os.path.expanduser("~"))


def test_set_seed_repeatable():
    set_seed(123)
    first = torch.rand(4)
    set_seed(123)
    assert torch.equal(first, torch.rand(4))
