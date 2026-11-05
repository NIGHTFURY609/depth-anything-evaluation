"""Evaluates cached predictions against both ground truth caches.

Per-image rows go to results/phase2/<split>/*.csv.gz (git-ignored, rebuilt from the caches);
bootstrapped summaries and Markdown tables go next to them and are committed. Any model whose
predictions are cached under <cache_root>/pred/<name>/ can be evaluated, so fine-tuned models
need no change here.
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
from src.eval import ALIGN_METHODS, check_can_evaluate, evaluate_frame
from src.gt import frame_path
from src.infer import eval_frames
from src.report import fmt, paired, summarize, within_model_differences

GT_SOURCES = ["gt", "gt_sensitivity"]
ACCURACY = ["absrel", "rmse", "delta1", "phantom", "overshoot", "clamped"]
PRIMARY_ALIGN = {"depth": "median", "disparity": "ls"}
TABLE_SUBSETS = ["all", "grass", "trees_bushes", "vegetation", "ground", "other", "edge", "edge_depth",
                 "edge_semantic"]


def _evaluate(task, cfg):
    model, gt_source, pos, seq, frame = task
    cache = cfg["paths"]["cache_root"]
    with np.load(frame_path(cache, seq, frame, f"pred/{model}")) as p:
        pred, kind, model_id = p["pred"], str(p["kind"]), str(p["model_id"])
        conf = p["conf"] if "conf" in p.files else None
        cached_hash = str(p["git_hash"])
    expected_id = cfg["models"].get(model)
    if expected_id is not None and model_id != expected_id:
        raise SystemExit(f"pred/{model} holds {model_id}, but config.yaml says {expected_id}")
    with np.load(frame_path(cache, seq, frame, gt_source)) as g:
        gt = dict(g)
    if pred.shape != gt["depth"].shape:
        raise SystemExit(f"pred/{model} {seq}/{frame}: shape {pred.shape} != GT {gt['depth'].shape}")
    rng = np.random.default_rng([cfg["seed"], int(seq), frame])
    out = evaluate_frame(pred, conf, kind, gt, cfg["eval"], rng)
    tag = {"seq": seq, "frame": frame, "pos": pos, "model": model, "kind": kind, "gt_source": gt_source}
    for name in ("accuracy", "confidence", "filtering", "curves"):
        out[name] = [{**tag, **row} for row in out[name]]
    out["skipped"] = [tag] if out["skipped"] else []
    out["provenance"] = (model, model_id, cached_hash)
    return out


def run(cfg, split: str, models: list[str], gt_sources: list[str]):
    frames = pd.read_csv(Path(cfg["paths"]["results_root"]) / "phase2" / f"frames_{split}.csv",
                         dtype={"seq": str})
    expected = [(f.seq, f.frame) for f in eval_frames(cfg, split)]
    if list(zip(frames.seq, frames.frame)) != expected:
        raise SystemExit(f"frames_{split}.csv does not match the config's frame selection")
    tasks = [(m, g, i, s, f) for m in models for g in gt_sources for i, (s, f) in enumerate(expected)]
    collected = {k: [] for k in ("accuracy", "confidence", "filtering", "curves", "skipped")}
    provenance = set()
    # close and join rather than the context manager's terminate: pycolmap, imported through
    # DA3, prints a stack trace when a worker receives SIGTERM.
    pool = Pool(8)
    for out in tqdm(pool.imap(partial(_evaluate, cfg=cfg), tasks, chunksize=8), total=len(tasks)):
        for k in collected:
            collected[k] += out[k]
        provenance.add(out["provenance"])
    pool.close()
    pool.join()
    return {k: pd.DataFrame(v) for k, v in collected.items()}, len(expected), sorted(provenance)


def mean_curves(curves: pd.DataFrame, ecfg: dict) -> pd.DataFrame:
    fractions = np.arange(0, ecfg["sparsification_max"] + 1e-9, ecfg["sparsification_step"])
    rows = []
    for (model, gt_source, align), g in curves.groupby(["model", "gt_source", "align"]) if not curves.empty else []:
        for name in ("conf", "oracle"):
            mean = np.mean(np.stack(g[name].to_numpy()), axis=0)
            rows += [{"model": model, "gt_source": gt_source, "align": align, "curve": name,
                      "fraction": round(f, 2), "absrel": v} for f, v in zip(fractions, mean)]
    return pd.DataFrame(rows, columns=["model", "gt_source", "align", "curve", "fraction", "absrel"])


def filter_pairs(ecfg: dict) -> list[tuple[str, str]]:
    """Registered filter comparisons, each as (a, b) for the per-image difference a - b."""
    pairs = []
    for b in ecfg["filter_budgets"]:
        pairs += [(f"confrank{b:g}", f"sobelrank{b:g}"), (f"confrank{b:g}", f"randomrank{b:g}"),
                  (f"sobelrank{b:g}", f"randomrank{b:g}")]
    for th in ecfg["sobel_thickness"]:
        pairs += [(f"conf_valid{th}", f"sobel{th}"), (f"sobel{th}", f"random_valid{th}")]
    return pairs


def _cell(summary: pd.DataFrame, n_frames: int, **match) -> str:
    row = summary
    for k, v in match.items():
        row = row[row[k] == v]
    if row.empty:
        return "-"
    if len(row) > 1:
        raise ValueError(f"{len(row)} summary rows match {match}")
    r = row.iloc[0]
    return fmt(r["mean"], r["lo"], r["hi"]) + (f" (n={r['n_images']})" if r["n_images"] < 0.9 * n_frames else "")


def tables(split, models, kinds, acc_s, conf_s, filt_s, paired_s, conf_paired_s, diff_s, ecfg,
           n_frames) -> str:
    """Markdown tables 1, 3 and 4 (zero-shot parts), for every GT source and alignment."""
    cell = partial(_cell, n_frames=n_frames)
    conf_models = [m for m in models if kinds[m] == "depth"]
    budgets = [f"{b:g}" for b in ecfg["filter_budgets"]]
    out = [f"# Phase 2 results: {split}\n",
           "Mean over images with 95% block-bootstrap intervals. `(n=...)` marks cells with under 90% "
           "of images, where the subset was too small. Alignment: `median` scale (depth models), "
           "`ls` / `huber` scale and shift in disparity (DA2) or inverse depth (DA3). Primary "
           "alignment: `median` for depth models, `ls` for disparity models.\n"]
    for gt_source in GT_SOURCES:
        for metric in ("absrel", "delta1", "phantom"):
            out.append(f"\n## Table 1: {metric}, LiDAR labels, {gt_source}\n")
            out.append("| model | align | " + " | ".join(TABLE_SUBSETS) + " |")
            out.append("|" + " --- |" * (len(TABLE_SUBSETS) + 2))
            for model in models:
                for align in ALIGN_METHODS[kinds[model]]:
                    cells = [cell(acc_s, model=model, gt_source=gt_source, label_source="lidar", align=align,
                                  subset=s, metric=metric) for s in TABLE_SUBSETS]
                    out.append(f"| {model} | {align} | " + " | ".join(cells) + " |")

        out.append(f"\n## Table 1b: AbsRel by distance, LiDAR labels, primary alignment, {gt_source}\n")
        bins = [f"{a}-{b}m" for a, b in zip(ecfg["distance_bins"][:-1], ecfg["distance_bins"][1:])]
        cols = [f"{g}|{b}" for g in ("all", "grass", "trees_bushes", "ground") for b in bins]
        out.append("| model | " + " | ".join(cols) + " |")
        out.append("|" + " --- |" * (len(cols) + 1))
        for model in models:
            cells = [cell(acc_s, model=model, gt_source=gt_source, label_source="lidar",
                          align=PRIMARY_ALIGN[kinds[model]], subset=c, metric="absrel") for c in cols]
            out.append(f"| {model} | " + " | ".join(cells) + " |")

        out.append(f"\n## Table 1c: AbsRel, LiDAR vs image labels on annotated frames, primary alignment, {gt_source}\n")
        subsets = ["grass", "trees_bushes", "vegetation", "ground"]
        out.append("| model | labels | " + " | ".join(subsets) + " |")
        out.append("|" + " --- |" * (len(subsets) + 2))
        for model in models:
            for source in ("lidar_annotated", "image"):
                cells = [cell(acc_s, model=model, gt_source=gt_source, label_source=source,
                              align=PRIMARY_ALIGN[kinds[model]], subset=s, metric="absrel") for s in subsets]
                out.append(f"| {model} | {source} | " + " | ".join(cells) + " |")

        for metric in ("absrel", "phantom"):
            out.append(f"\n## Paired {metric} difference against {ecfg['reference_model']}, LiDAR labels, {gt_source}\n")
            out.append("Positive means higher than the reference. Same alignment on both sides.\n")
            out.append("| model | align | " + " | ".join(TABLE_SUBSETS) + " |")
            out.append("|" + " --- |" * (len(TABLE_SUBSETS) + 2))
            for (model, align), _ in paired_s.groupby(["model", "align"]) if not paired_s.empty else []:
                cells = [cell(paired_s, model=model, gt_source=gt_source, label_source="lidar", align=align,
                              subset=s, metric=metric) for s in TABLE_SUBSETS]
                out.append(f"| {model} | {align} | " + " | ".join(cells) + " |")

        out.append(f"\n## Table 3: confidence quality, {gt_source}\n")
        out.append(f"| model | align | AUROC (AbsRel > {ecfg['high_error']}) | AUSE | AURG |")
        out.append("| --- | --- | --- | --- | --- |")
        for model in conf_models:
            for align in ALIGN_METHODS["depth"]:
                cells = [cell(conf_s, model=model, gt_source=gt_source, align=align, metric=m)
                         for m in ("auroc", "ause", "aurg")]
                out.append(f"| {model} | {align} | " + " | ".join(cells) + " |")
        if not conf_paired_s.empty:
            out.append(f"\nPaired difference against {ecfg['reference_model']}:\n")
            out.append("| model | align | AUROC | AUSE | AURG |")
            out.append("| --- | --- | --- | --- | --- |")
            for (model, align), _ in conf_paired_s.groupby(["model", "align"]):
                cells = [cell(conf_paired_s, model=model, gt_source=gt_source, align=align, metric=m)
                         for m in ("auroc", "ause", "aurg")]
                out.append(f"| {model} | {align} | " + " | ".join(cells) + " |")

        out.append(f"\n## Table 4: filtering at fixed budgets (primary), primary alignment, {gt_source}\n")
        out.append("AbsRel and phantom rate on the valid pixels left after removing a fixed share of them: "
                   f"highest Sobel score (kernel {ecfg['sobel_rank_thickness']}), lowest confidence, or at "
                   "random. An image where any filter leaves under "
                   f"{ecfg['min_subset_px']} pixels is dropped from that budget for every filter.\n")
        out.append("| model | budget | metric | none | sobelrank | confrank | randomrank | confrank - sobelrank |")
        out.append("|" + " --- |" * 8)
        for model in models:
            align = PRIMARY_ALIGN[kinds[model]]
            for b in budgets:
                for metric in ("absrel", "phantom"):
                    cells = [cell(filt_s, model=model, gt_source=gt_source, align=align, filter=f, metric=metric)
                             for f in ("none", f"sobelrank{b}", f"confrank{b}", f"randomrank{b}")]
                    cells.append(cell(diff_s, model=model, gt_source=gt_source, align=align,
                                      filter=f"confrank{b}-sobelrank{b}", metric=metric))
                    out.append(f"| {model} | {b} | {metric} | " + " | ".join(cells) + " |")

        out.append(f"\n## Table 4b: the Offroad-Nav mask at its own budget, primary alignment, {gt_source}\n")
        out.append("Sobel with Otsu removes most valid pixels on RELLIS. `conf_valid` removes as many valid "
                   "pixels as Sobel did; `conf_all` as many pixels of the whole image.\n")
        out.append("| model | thickness | Sobel removes (valid) | metric | none | sobel | conf_valid | conf_all | random_valid | conf_valid - sobel |")
        out.append("|" + " --- |" * 10)
        for model in models:
            align = PRIMARY_ALIGN[kinds[model]]
            for th in ecfg["sobel_thickness"]:
                removed = cell(filt_s, model=model, gt_source=gt_source, align=align,
                               filter=f"sobel{th}", metric="removed_valid")
                for metric in ("absrel", "phantom"):
                    cells = [cell(filt_s, model=model, gt_source=gt_source, align=align, filter=f, metric=metric)
                             for f in ("none", f"sobel{th}", f"conf_valid{th}", f"conf_all{th}", f"random_valid{th}")]
                    cells.append(cell(diff_s, model=model, gt_source=gt_source, align=align,
                                      filter=f"conf_valid{th}-sobel{th}", metric=metric))
                    out.append(f"| {model} | {th} | {removed} | {metric} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--models", nargs="+", help="names under pred/; defaults to config models")
    parser.add_argument("--out", help="output directory; defaults to results/phase2/<split>")
    args = parser.parse_args()

    cfg = load_config()
    ecfg = cfg["eval"]
    models = args.models or list(cfg["models"])
    commit = git_hash()
    check_can_evaluate(args.split, ecfg["frozen"], commit.endswith("-dirty"))

    data, n_frames, provenance = run(cfg, args.split, models, GT_SOURCES)
    out_dir = Path(args.out) if args.out else Path(cfg["paths"]["results_root"]) / "phase2" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("accuracy", "confidence", "filtering"):
        data[name].to_csv(out_dir / f"{name}_per_image.csv.gz", index=False)

    b = ecfg["bootstrap"]
    boot = (b["block"], b["resamples"], b["seed"])
    acc = data["accuracy"]
    kinds = acc.groupby("model")["kind"].first().to_dict()
    annotated = acc[acc["label_source"] == "image"][["seq", "frame"]].drop_duplicates()
    lidar_annotated = acc[acc["label_source"] == "lidar"].merge(annotated, on=["seq", "frame"])
    acc = pd.concat([acc, lidar_annotated.assign(label_source="lidar_annotated")])
    keys = ["model", "gt_source", "label_source", "align", "subset"]
    acc_s = summarize(acc, keys, ACCURACY, *boot)
    conf_keys = ["model", "gt_source", "align"]
    conf_s = summarize(data["confidence"], conf_keys, ["auroc", "ause", "aurg"], *boot)
    filt = data["filtering"]
    filt_keys = ["model", "gt_source", "align", "filter"]
    filt_s = summarize(filt, filt_keys, ["absrel", "phantom", "removed_valid", "removed_all"], *boot)
    diffs = within_model_differences(filt, ["model", "gt_source", "align"], filter_pairs(ecfg),
                                     ["absrel", "phantom"])
    diff_s = summarize(diffs, filt_keys, ["absrel", "phantom"], *boot)
    lidar = acc[(acc["label_source"] == "lidar") & acc["subset"].isin(TABLE_SUBSETS)]
    ref = ecfg["reference_model"]
    paired_s = paired(lidar, ["gt_source", "label_source", "align", "subset"], ["absrel", "phantom"], ref, *boot)
    conf_paired_s = (paired(data["confidence"], ["gt_source", "align"], ["auroc", "ause", "aurg"], ref, *boot)
                     if not data["confidence"].empty else pd.DataFrame())
    for name, df in (("accuracy", acc_s), ("confidence", conf_s), ("filtering", filt_s),
                     ("filter_differences", diff_s), ("paired", paired_s),
                     ("paired_confidence", conf_paired_s), ("curves", mean_curves(data["curves"], ecfg))):
        df.to_csv(out_dir / f"summary_{name}.csv", index=False, float_format="%.6g")

    (out_dir / "tables.md").write_text(tables(args.split, models, kinds, acc_s, conf_s, filt_s, paired_s,
                                              conf_paired_s, diff_s, ecfg, n_frames))
    meta = {
        "split": args.split, "git_hash": commit, "models": models, "gt_sources": GT_SOURCES,
        "config_sha256": hashlib.sha256(Path("config.yaml").read_bytes()).hexdigest(),
        "frames": n_frames, "skipped": data["skipped"].to_dict("records"),
        "predictions": [{"model": m, "model_id": i, "cached_at": h} for m, i, h in provenance],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_dir}; skipped {len(data['skipped'])} frame evaluations")


if __name__ == "__main__":
    main()
