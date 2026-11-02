import os
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.verify_rellis import check_sequences, is_valid_zip, main


def make_frame(root: Path, seq: str, frame: int, scan=True, label=True) -> None:
    images = root / seq / "pylon_camera_node"
    scans = root / seq / "os1_cloud_node_kitti_bin"
    labels = root / seq / "os1_cloud_node_semantickitti_label_id"
    for folder in (images, scans, labels):
        folder.mkdir(parents=True, exist_ok=True)
    (images / f"frame{frame:06d}-1581624663_149.jpg").touch()
    if scan:
        (scans / f"{frame:06d}.bin").touch()
    if label:
        (labels / f"{frame:06d}.label").touch()


def test_counts_match(tmp_path):
    make_frame(tmp_path, "00000", 1)
    make_frame(tmp_path, "00000", 2)
    report = check_sequences(tmp_path)["00000"]
    assert (report["images"], report["scans"], report["labels"]) == (2, 2, 2)
    assert report["images_without_scan"] == []
    assert report["scans_without_label"] == []


def test_image_without_scan_reported(tmp_path):
    make_frame(tmp_path, "00000", 1)
    make_frame(tmp_path, "00000", 2, scan=False, label=False)
    assert check_sequences(tmp_path)["00000"]["images_without_scan"] == [2]


def test_missing_label_reported(tmp_path):
    make_frame(tmp_path, "00000", 1)
    make_frame(tmp_path, "00000", 2, label=False)
    assert check_sequences(tmp_path)["00000"]["scans_without_label"] == [2]


def test_html_quota_page_is_not_valid_zip(tmp_path):
    page = tmp_path / "scans.zip"
    page.write_text("<!DOCTYPE html><html>Google Drive quota exceeded</html>")
    assert is_valid_zip(page) is False


def test_invalid_download_has_its_own_exit_code(tmp_path, monkeypatch):
    page = tmp_path / "scans.zip"
    page.write_text("<!DOCTYPE html><html>quota</html>")
    monkeypatch.setattr("sys.argv", ["verify_rellis.py", "--check-download", str(page)])
    assert main() == 3


def run_download_with_fake_python(tmp_path: Path, exit_code: int):
    """Runs the download script with a stand-in for Python that always exits with exit_code."""
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(f"#!/bin/sh\nexit {exit_code}\n")
    fake_python.chmod(0o755)
    existing = tmp_path / "raw" / "ontology" / "ontology.zip"
    existing.parent.mkdir(parents=True)
    with zipfile.ZipFile(existing, "w") as z:
        z.writestr("ontology.yaml", "ok")
    env = {**os.environ, "RELLIS_RAW": str(tmp_path / "raw"), "PY": str(fake_python)}
    result = subprocess.run(
        ["bash", "scripts/download_rellis.sh", "small"], env=env, capture_output=True, text=True
    )
    return result, existing


@pytest.mark.parametrize("crash_code", [1, 2, 127])
def test_broken_checker_keeps_existing_download(tmp_path, crash_code):
    result, existing = run_download_with_fake_python(tmp_path, crash_code)
    assert existing.exists()
    assert result.returncode != 0
    assert "could not check" in result.stdout


def test_missing_python_keeps_existing_download(tmp_path):
    existing = tmp_path / "raw" / "ontology" / "ontology.zip"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"data")
    env = {**os.environ, "RELLIS_RAW": str(tmp_path / "raw"), "PY": str(tmp_path / "nope")}
    result = subprocess.run(
        ["bash", "scripts/download_rellis.sh", "small"], env=env, capture_output=True, text=True
    )
    assert existing.exists()
    assert result.returncode != 0


def test_invalid_download_is_still_deleted(tmp_path):
    result, existing = run_download_with_fake_python(tmp_path, 3)
    assert not existing.exists()
    assert "FAILED ontology" in result.stdout
    assert "drive.google.com/file/d/" in result.stdout
