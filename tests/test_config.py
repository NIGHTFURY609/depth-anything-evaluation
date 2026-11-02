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


def test_split_is_by_whole_sequence_and_covers_all():
    cfg = load_config()
    split = cfg["split"]
    train, val, test = set(split["train"]), {split["val"]}, {split["test"]}
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(cfg["gt"]["sequences"])
    assert len(train) == 3


def test_set_seed_repeatable():
    set_seed(123)
    first = torch.rand(4)
    set_seed(123)
    assert torch.equal(first, torch.rand(4))
