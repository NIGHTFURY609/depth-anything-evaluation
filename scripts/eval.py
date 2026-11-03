"""Evaluates cached predictions against both ground truth caches.

Per-image rows go to results/phase2/<split>/*.csv.gz (git-ignored, rebuilt from the caches);
bootstrapped summaries and Markdown tables go next to them and are committed.
"""

import argparse
import hashlib
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import git_hash, load_config
from src.eval import check_can_evaluate, evaluate_frame
from src.gt import frame_path
from src.infer import eval_frames
from src.report import fmt, paired, summarize

GT_SOURCES = ["gt", "gt_sensitivity"]
MODELS = ["da2_small", "da2_base", "da3_small", "da3_base"]
ACCURACY = ["absrel", "rmse", "delta1", "phantom", "overshoot", "clamped"]
PRIMARY_ALIGN = {"da2_small": "ls", "da2_base": "ls", "da3_small": "median", "da3_base": "median"}
TABLE_SUBSETS = ["all", "grass", "trees_bushes", "vegetation", "ground", "other", "edge", "edge_depth",
                 "edge_semantic"]


def _evaluate(task, cfg):
    model, gt_source, seq, frame = task
    cache = cfg["paths"]["cache_root"]
    with np.load(frame_path(cache, seq, frame, f"pred/{model}")) as p:
        pred, kind = p["pred"], str(p["kind"])
        conf = p["conf"] if "conf" in p.files else None
    with np.load(frame_path(cache, seq, frame, gt_source)) as g:
        gt = dict(g)
    rng = np.random.default_rng([cfg["seed"], int(seq), frame])
    out = evaluate_frame(pred, conf, kind, gt, cfg["eval"], rng)
    tag = {"seq": seq, "frame": frame, "model": model, "gt_source": gt_source}
    for name in ("accuracy", "confidence", "filtering", "curves"):
        out[name] = [{**tag, **row} for row in out[name]]
    out["skipped"] = [tag] if out["skipped"] else []
    return out


def run(cfg, split: str, models: list[str], gt_sources: list[str]) -> dict[str, pd.DataFrame]:
    frames = pd.read_csv(Path(cfg["paths"]["results_root"]) / "phase2" / f"frames_{split}.csv",
                         dtype={"seq": str})
    expected = [(f.seq, f.frame) for f in eval_frames(cfg, split)]
    if list(zip(frames.seq, frames.frame)) != expected:
        raise SystemExit(f"frames_{split}.csv does not match the config's frame selection")
    tasks = [(m, g, s, f) for m in models for g in gt_sources for s, f in expected]
    collected = {k: [] for k in ("accuracy", "confidence", "filtering", "curves", "skipped")}
    # close and join rather than the context manager's terminate: pycolmap, imported through
    # DA3, prints a stack trace when a worker receives SIGTERM.
    pool = Pool(8)
    for out in tqdm(pool.imap(partial(_evaluate, cfg=cfg), tasks, chunksize=8), total=len(tasks)):
        for k in collected:
            collected[k] += out[k]
    pool.close()
    pool.join()
    return {k: pd.DataFrame(v) for k, v in collected.items()}


def mean_curves(curves: pd.DataFrame, ecfg: dict) -> pd.DataFrame:
    fractions = np.arange(0, ecfg["sparsification_max"] + 1e-9, ecfg["sparsification_step"])
    rows = []
    for (model, gt_source, align), g in curves.groupby(["model", "gt_source", "align"]):
        for name in ("conf", "oracle"):
            mean = np.mean(np.stack(g[name].to_numpy()), axis=0)
            rows += [{"model": model, "gt_source": gt_source, "align": align, "curve": name,
                      "fraction": round(f, 2), "absrel": v} for f, v in zip(fractions, mean)]
    return pd.DataFrame(rows)


def _cell(summary: pd.DataFrame, n_frames: int, **match) -> str:
    row = summary
    for k, v in match.items():
        row = row[row[k] == v]
    if row.empty:
        return "-"
    r = row.iloc[0]
    return fmt(r["mean"], r["lo"], r["hi"]) + (f" (n={r['n_images']})" if r["n_images"] < 0.9 * n_frames else "")


def tables(split: str, acc_s, conf_s, filt_s, paired_s, ecfg, n_frames: int) -> str:
    """Markdown tables 1, 3 and 4 (zero-shot parts), for every GT source and alignment."""
    cell = partial(_cell, n_frames=n_frames)
    out = [f"# Phase 2 results: {split}\n",
           "Mean over images with 95% block-bootstrap intervals. `(n=...)` marks cells with under 90% "
           "of images, where the subset was too small. Alignment: `median` scale (DA3), `ls` / `huber` "
           "scale and shift in disparity (DA2) or inverse depth (DA3).\n"]
    for gt_source in GT_SOURCES:
        for metric in ("absrel", "delta1", "phantom"):
            out.append(f"\n## Table 1: {metric}, LiDAR labels, {gt_source}\n")
            out.append("| model | align | " + " | ".join(TABLE_SUBSETS) + " |")
            out.append("|" + " --- |" * (len(TABLE_SUBSETS) + 2))
            for model in MODELS:
                for align in sorted(acc_s[acc_s.model == model]["align"].unique()):
                    cells = [cell(acc_s, model=model, gt_source=gt_source, label_source="lidar", align=align,
                                  subset=s, metric=metric) for s in TABLE_SUBSETS]
                    out.append(f"| {model} | {align} | " + " | ".join(cells) + " |")

        out.append(f"\n## Table 1b: AbsRel by distance, LiDAR labels, primary alignment, {gt_source}\n")
        bins = [f"{a}-{b}m" for a, b in zip(ecfg["distance_bins"][:-1], ecfg["distance_bins"][1:])]
        cols = [f"{g}|{b}" for g in ("all", "grass", "trees_bushes", "ground") for b in bins]
        out.append("| model | " + " | ".join(cols) + " |")
        out.append("|" + " --- |" * (len(cols) + 1))
        for model in MODELS:
            cells = [cell(acc_s, model=model, gt_source=gt_source, label_source="lidar",
                          align=PRIMARY_ALIGN[model], subset=c, metric="absrel") for c in cols]
            out.append(f"| {model} | " + " | ".join(cells) + " |")

        out.append(f"\n## Table 1c: AbsRel, LiDAR vs image labels on annotated frames, primary alignment, {gt_source}\n")
        subsets = ["grass", "trees_bushes", "vegetation", "ground"]
        out.append("| model | labels | " + " | ".join(subsets) + " |")
        out.append("|" + " --- |" * (len(subsets) + 2))
        for model in MODELS:
            for source in ("lidar_annotated", "image"):
                cells = [cell(acc_s, model=model, gt_source=gt_source, label_source=source,
                              align=PRIMARY_ALIGN[model], subset=s, metric="absrel") for s in subsets]
                out.append(f"| {model} | {source} | " + " | ".join(cells) + " |")

        out.append(f"\n## Paired AbsRel difference against {ecfg['reference_model']}, LiDAR labels, {gt_source}\n")
        out.append("Positive means worse than the reference. Same alignment on both sides.\n")
        out.append("| model | align | " + " | ".join(TABLE_SUBSETS) + " |")
        out.append("|" + " --- |" * (len(TABLE_SUBSETS) + 2))
        for (model, align), _ in paired_s.groupby(["model", "align"]):
            cells = [cell(paired_s, model=model, gt_source=gt_source, label_source="lidar", align=align,
                          subset=s, metric="absrel") for s in TABLE_SUBSETS]
            out.append(f"| {model} | {align} | " + " | ".join(cells) + " |")

        out.append(f"\n## Table 3: confidence quality, {gt_source}\n")
        out.append("| model | align | AUROC (AbsRel > 0.25) | AUSE | AURG |")
        out.append("| --- | --- | --- | --- | --- |")
        for model in ("da3_small", "da3_base"):
            for align in ("median", "ls", "huber"):
                cells = [cell(conf_s, model=model, gt_source=gt_source, align=align, metric=m)
                         for m in ("auroc", "ause", "aurg")]
                out.append(f"| {model} | {align} | " + " | ".join(cells) + " |")

        out.append(f"\n## Table 4: equal-budget filtering, primary alignment, {gt_source}\n")
        out.append("AbsRel and phantom rate on the valid pixels left. `conf_valid` removes as many valid "
                   "pixels as Sobel did; `conf_all` as many pixels of the whole image.\n")
        out.append("| model | thickness | Sobel removes (valid) | metric | none | sobel | conf_valid | conf_all | random_valid |")
        out.append("|" + " --- |" * 9)
        for model in MODELS:
            for th in ecfg["sobel_thickness"]:
                removed = cell(filt_s, model=model, gt_source=gt_source, align=PRIMARY_ALIGN[model],
                               filter=f"sobel{th}", metric="removed_valid")
                for metric in ("absrel", "phantom"):
                    cells = [cell(filt_s, model=model, gt_source=gt_source, align=PRIMARY_ALIGN[model],
                                  filter=f, metric=metric)
                             for f in ("none", f"sobel{th}", f"conf_valid{th}", f"conf_all{th}", f"random_valid{th}")]
                    out.append(f"| {model} | {th} | {removed} | {metric} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    args = parser.parse_args()

    cfg = load_config()
    ecfg = cfg["eval"]
    commit = git_hash()
    check_can_evaluate(args.split, ecfg["frozen"], commit.endswith("-dirty"))

    data = run(cfg, args.split, args.models, GT_SOURCES)
    out_dir = Path(cfg["paths"]["results_root"]) / "phase2" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("accuracy", "confidence", "filtering"):
        data[name].to_csv(out_dir / f"{name}_per_image.csv.gz", index=False)

    b = ecfg["bootstrap"]
    boot = (b["block"], b["resamples"], b["seed"])
    acc = data["accuracy"]
    annotated = acc[acc.label_source == "image"][["seq", "frame"]].drop_duplicates()
    lidar_annotated = acc[acc.label_source == "lidar"].merge(annotated, on=["seq", "frame"])
    acc = pd.concat([acc, lidar_annotated.assign(label_source="lidar_annotated")])
    keys = ["model", "gt_source", "label_source", "align", "subset"]
    acc_s = summarize(acc, keys, ACCURACY, *boot)
    conf_s = summarize(data["confidence"], ["model", "gt_source", "align"], ["auroc", "ause", "aurg"], *boot)
    filt_s = summarize(data["filtering"], ["model", "gt_source", "align", "filter"],
                       ["absrel", "phantom", "removed_valid", "removed_all"], *boot)
    main_subsets = acc[acc.label_source == "lidar"]
    paired_s = paired(main_subsets[main_subsets.subset.isin(TABLE_SUBSETS)],
                      ["gt_source", "label_source", "align", "subset"], ["absrel"],
                      ecfg["reference_model"], *boot)
    for name, df in (("accuracy", acc_s), ("confidence", conf_s), ("filtering", filt_s),
                     ("paired", paired_s), ("curves", mean_curves(data["curves"], ecfg))):
        df.to_csv(out_dir / f"summary_{name}.csv", index=False, float_format="%.6g")

    n_frames = len(pd.read_csv(Path(cfg["paths"]["results_root"]) / "phase2" / f"frames_{args.split}.csv"))
    (out_dir / "tables.md").write_text(tables(args.split, acc_s, conf_s, filt_s, paired_s, ecfg, n_frames))
    meta = {
        "split": args.split, "git_hash": commit, "models": args.models, "gt_sources": GT_SOURCES,
        "config_sha256": hashlib.sha256(Path("config.yaml").read_bytes()).hexdigest(),
        "frames": n_frames, "skipped": data["skipped"].to_dict("records"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_dir}; skipped {len(data['skipped'])} frame evaluations")


if __name__ == "__main__":
    main()
