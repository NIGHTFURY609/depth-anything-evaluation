"""Checks the extracted RELLIS-3D tree, or a single downloaded file with --check-download."""

import argparse
import re
import sys
import zipfile
from pathlib import Path

FRAME_RE = re.compile(r"frame(\d+)-")


def is_valid_zip(path: Path) -> bool:
    return zipfile.is_zipfile(path)


def is_valid_download(path: Path) -> bool:
    """A Drive quota page arrives as HTML, sometimes under the expected file name."""
    path = Path(path)
    if path.suffix == ".zip":
        return is_valid_zip(path)
    head = path.read_bytes()[:256].lstrip().lower()
    return path.stat().st_size > 0 and not head.startswith((b"<!doctype html", b"<html"))


def _frame_numbers(folder: Path, pattern: str) -> set[int]:
    return {int(p.stem) for p in folder.glob(pattern)} if folder.is_dir() else set()


def check_sequences(root: Path) -> dict[str, dict]:
    report = {}
    for seq_dir in sorted(p for p in Path(root).iterdir() if p.is_dir() and p.name.isdigit()):
        image_dir = seq_dir / "pylon_camera_node"
        images = set()
        if image_dir.is_dir():
            for path in image_dir.glob("frame*.jpg"):
                match = FRAME_RE.match(path.name)
                if match:
                    images.add(int(match.group(1)))
        scans = _frame_numbers(seq_dir / "os1_cloud_node_kitti_bin", "*.bin")
        labels = _frame_numbers(seq_dir / "os1_cloud_node_semantickitti_label_id", "*.label")
        report[seq_dir.name] = {
            "images": len(images),
            "scans": len(scans),
            "labels": len(labels),
            "images_without_scan": sorted(images - scans),
            "scans_without_label": sorted(scans - labels),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-download", type=Path)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if args.check_download:
        return 0 if is_valid_download(args.check_download) else 1

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import load_config

    cfg = load_config(args.config)
    expected = cfg["rellis"]["expected_images"]
    report = check_sequences(Path(cfg["paths"]["data_root"]) / "Rellis-3D")

    ok = set(report) == set(expected)
    for seq, want in expected.items():
        row = report.get(seq)
        if row is None:
            print(f"{seq}: MISSING")
            continue
        good = (
            row["images"] == want
            and row["scans"] == want
            and row["labels"] == want
            and not row["images_without_scan"]
            and not row["scans_without_label"]
        )
        ok = ok and good
        print(
            f"{seq}: images={row['images']}/{want} scans={row['scans']} labels={row['labels']} "
            f"images_without_scan={len(row['images_without_scan'])} "
            f"scans_without_label={len(row['scans_without_label'])} {'ok' if good else 'MISMATCH'}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
