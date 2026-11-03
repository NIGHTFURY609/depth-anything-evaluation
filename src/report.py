"""Turns per-image rows into bootstrapped summaries and paired differences.

Rows carry `pos`, the image's index in the split's frame list, when it is known; blocks are then
cut on it rather than on row order.
"""

import numpy as np
import pandas as pd

from src.eval import block_bootstrap

_STATS = ["metric", "mean", "lo", "hi", "n_images"]


def _positions(group: pd.DataFrame):
    return group["pos"].to_numpy() if "pos" in group else None


def summarize(df: pd.DataFrame, keys: list[str], metrics: list[str], block: int, resamples: int,
              seed: int) -> pd.DataFrame:
    """Mean and 95% block-bootstrap interval of each metric per group; images in frame order."""
    rows = []
    for key, group in df.groupby(keys, sort=True) if not df.empty else []:
        group = group.sort_values(["seq", "frame"])
        key = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            values = group[metric].to_numpy(np.float64)
            mean, lo, hi = block_bootstrap(values, block, resamples, seed, _positions(group))
            rows.append({**dict(zip(keys, key)), "metric": metric, "mean": mean, "lo": lo, "hi": hi,
                         "n_images": int(np.isfinite(values).sum())})
    return pd.DataFrame(rows, columns=keys + _STATS)


def paired(df: pd.DataFrame, keys: list[str], metrics: list[str], reference: str, block: int,
           resamples: int, seed: int) -> pd.DataFrame:
    """Each model minus the reference model, over images where both have a value."""
    ref = df[df["model"] == reference]
    rows = []
    for model in sorted(set(df["model"]) - {reference}):
        both = df[df["model"] == model].merge(ref, on=keys + ["seq", "frame"], suffixes=("", "_ref"))
        for key, group in both.groupby(keys, sort=True):
            group = group.sort_values(["seq", "frame"])
            key = key if isinstance(key, tuple) else (key,)
            for metric in metrics:
                diff = group[metric].to_numpy(np.float64) - group[f"{metric}_ref"].to_numpy(np.float64)
                mean, lo, hi = block_bootstrap(diff, block, resamples, seed, _positions(group))
                rows.append({"model": model, "reference": reference, **dict(zip(keys, key)),
                             "metric": metric, "mean": mean, "lo": lo, "hi": hi,
                             "n_images": int(np.isfinite(diff).sum())})
    return pd.DataFrame(rows, columns=["model", "reference", *keys, *_STATS])


def within_model_differences(df: pd.DataFrame, keys: list[str], pairs: list[tuple[str, str]],
                             metrics: list[str]) -> pd.DataFrame:
    """Per-image differences between two filters of the same model, as rows named "a-b", ready
    for `summarize`. Images where either side is NaN stay NaN."""
    index = ["seq", "frame", *(["pos"] if "pos" in df else []), *keys]
    out = []
    for a, b in pairs:
        left = df[df["filter"] == a].set_index(index)[metrics]
        right = df[df["filter"] == b].set_index(index)[metrics]
        diff = (left - right.reindex(left.index)).reset_index()
        out.append(diff.assign(filter=f"{a}-{b}"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=index + metrics)


def fmt(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "-"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
