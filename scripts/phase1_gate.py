"""Phase 1 gate: overlays, label agreement (with deliberately wrong controls), class counts.

Writes results/phase1_agreement.csv, results/phase1_controls.csv, results/phase1_shift.csv,
results/phase1_class_counts.csv and results/overlays/ (not committed: derived from RELLIS images).
"""

import csv
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.gt import build_frame, frame_path, load_calibration
from src.projection import label_agreement, shift_agreement
from src.rellis import GROUP_NAMES, list_frames, terrain_groups

OVERLAYS_PER_SEQUENCE = 4
CONTROL_FRAMES_PER_SEQUENCE = 40
CONTROL_FRAME_OFFSET = 10  # About one second of motion.
SHIFT_RADIUS = 6
SHIFT_FRAME_STRIDE = 5


def _spread(items: list, count: int) -> list:
    return [items[i] for i in np.linspace(0, len(items) - 1, count, dtype=int)]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_overlay(frame, gt: dict, cfg: dict, path: Path, scale: int = 2) -> None:
    """Kept points coloured by log depth; points removed by the occlusion filter in magenta."""
    width, height = cfg["input"]["width"], cfg["input"]["height"]
    image = cv2.imread(str(frame.image))
    image = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_AREA)
    depth = gt["depth"]
    log_depth = np.log(np.clip(depth, 1.0, 80.0)) / np.log(80.0)
    colours = cv2.applyColorMap((log_depth * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    for row, col in np.argwhere(depth > 0):
        centre = (int(col * scale), int(row * scale))
        colour = colours[row, col].tolist() if gt["keep"][row, col] else (255, 0, 255)
        cv2.circle(image, centre, 1, colour, -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])


def agreement_rows(frames: list, cfg: dict) -> list[dict]:
    rows = []
    for frame in tqdm(frames, desc=f"agreement {frames[0].seq}"):
        if frame.image_labels is None:
            continue
        with np.load(frame_path(cfg["paths"]["cache_root"], frame.seq, frame.frame)) as gt:
            labels, keep, image_labels = gt["labels"], gt["keep"], gt["image_labels"]
        rows.append(
            {
                "seq": frame.seq,
                "frame": frame.frame,
                "agreement_all": round(label_agreement(labels, image_labels), 4),
                "agreement_kept": round(label_agreement(labels * keep, image_labels), 4),
                "points": int(np.count_nonzero(labels)),
            }
        )
    return rows


def control_rows(frames: list, calib, cfg: dict) -> list[dict]:
    """The same frames scored under the correct setup and under deliberately wrong ones.

    If the metric could not tell these apart, a high score would prove nothing.
    """
    annotated = [f for f in frames if f.image_labels is not None]
    by_number = {f.frame: f for f in frames}
    sample = _spread(annotated, min(CONTROL_FRAMES_PER_SEQUENCE, len(annotated)))
    variants = {
        "correct": calib,
        "not_inverted": replace(calib, T_cam_lidar=np.linalg.inv(calib.T_cam_lidar)),
        "no_distortion": replace(calib, dist=np.zeros(5)),
    }
    scores = {name: [] for name in [*variants, f"scan_offset_{CONTROL_FRAME_OFFSET}"]}
    for frame in sample:
        image_labels = None
        for name, variant in variants.items():
            gt = build_frame(frame, variant, cfg)
            image_labels = gt["image_labels"]
            scores[name].append(label_agreement(gt["labels"], image_labels))
        other = by_number.get(frame.frame + CONTROL_FRAME_OFFSET)
        if other is not None:
            shifted = build_frame(replace(other, image_labels=None), calib, cfg)
            scores[f"scan_offset_{CONTROL_FRAME_OFFSET}"].append(
                label_agreement(shifted["labels"], image_labels)
            )
    return [
        {
            "seq": frames[0].seq,
            "variant": name,
            "frames": len(values),
            "median_agreement": round(float(np.nanmedian(values)), 4),
        }
        for name, values in scores.items()
    ]


def shift_row(frames: list, cfg: dict) -> dict:
    """Finds the pixel offset that best aligns LiDAR labels with image labels in this sequence."""
    annotated = [f for f in frames if f.image_labels is not None][::SHIFT_FRAME_STRIDE]
    grids = []
    for frame in tqdm(annotated, desc=f"shift {frames[0].seq}"):
        with np.load(frame_path(cfg["paths"]["cache_root"], frame.seq, frame.frame)) as gt:
            grids.append(shift_agreement(gt["labels"] * gt["keep"], gt["image_labels"], SHIFT_RADIUS))
    grid = np.nanmean(grids, axis=0)
    row, col = np.unravel_index(np.nanargmax(grid), grid.shape)
    return {
        "seq": frames[0].seq,
        "frames": len(annotated),
        "best_dx": int(col) - SHIFT_RADIUS,
        "best_dy": int(row) - SHIFT_RADIUS,
        "agreement_at_best": round(float(grid[row, col]), 4),
        "agreement_at_zero": round(float(grid[SHIFT_RADIUS, SHIFT_RADIUS]), 4),
    }


def class_count_row(frames: list, cfg: dict) -> dict:
    counts = {name: 0 for name in GROUP_NAMES.values()}
    for frame in tqdm(frames, desc=f"counts {frames[0].seq}"):
        with np.load(frame_path(cfg["paths"]["cache_root"], frame.seq, frame.frame)) as gt:
            groups = terrain_groups(gt["labels"][gt["keep"]])
        for group, name in GROUP_NAMES.items():
            counts[name] += int(np.count_nonzero(groups == group))
    labelled = sum(v for k, v in counts.items() if k != "void")
    return {
        "seq": frames[0].seq,
        "frames": len(frames),
        **counts,
        "grass_fraction": round(counts["grass"] / labelled, 4),
        "trees_bushes_fraction": round(counts["trees_bushes"] / labelled, 4),
    }


def main() -> None:
    cfg = load_config()
    results = Path(cfg["paths"]["results_root"])
    agreement, controls, counts, shifts = [], [], [], []
    for seq in cfg["gt"]["sequences"]:
        frames = list_frames(cfg["paths"]["data_root"], seq)
        calib = load_calibration(cfg, seq)
        for frame in _spread(frames, OVERLAYS_PER_SEQUENCE):
            draw_overlay(
                frame,
                build_frame(frame, calib, cfg),
                cfg,
                results / "overlays" / f"{seq}_{frame.frame:06d}.jpg",
            )
        agreement += agreement_rows(frames, cfg)
        controls += control_rows(frames, calib, cfg)
        counts.append(class_count_row(frames, cfg))
        shifts.append(shift_row(frames, cfg))
    write_csv(results / "phase1_shift.csv", shifts)
    write_csv(results / "phase1_agreement.csv", agreement)
    write_csv(results / "phase1_controls.csv", controls)
    write_csv(results / "phase1_class_counts.csv", counts)


if __name__ == "__main__":
    main()
