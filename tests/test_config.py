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


FREEZE_COMMIT = "6b486bc"


def test_eval_section_matches_freeze():
    """Phase 3 may add to config.yaml, but the evaluation protocol frozen at 6b486bc must not move."""
    import subprocess

    import yaml

    frozen = subprocess.run(["git", "show", f"{FREEZE_COMMIT}:config.yaml"], capture_output=True,
                            text=True, check=True).stdout
    frozen = yaml.safe_load(frozen)
    cfg = load_config()
    for section in ("eval", "split", "gt", "input", "models"):
        assert cfg[section] == frozen[section], section


def test_lambda_c_above_high_error_threshold():
    """c >= 1 and the optimum is lambda_c / e, so errors above lambda_c all sit at the floor."""
    cfg = load_config()
    assert cfg["train"]["lambda_c"] > cfg["eval"]["high_error"]


def test_git_hash_can_ignore_paths(tmp_path, monkeypatch):
    import subprocess

    from src.config import git_hash

    monkeypatch.chdir(tmp_path)
    run = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True)
    run("init", "-q")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "a.csv").write_text("1")
    assert git_hash().endswith("-dirty")
    assert not git_hash(ignore=("logs",)).endswith("-dirty")
    (tmp_path / "code.py").write_text("1")
    assert git_hash(ignore=("logs",)).endswith("-dirty")
