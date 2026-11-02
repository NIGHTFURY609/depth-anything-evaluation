from pathlib import Path

from scripts.verify_rellis import check_sequences, is_valid_zip


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
