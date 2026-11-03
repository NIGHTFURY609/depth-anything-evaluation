"""Turns per-image rows into bootstrapped summaries and paired differences."""

import numpy as np
import pandas as pd

from src.eval import block_bootstrap


def summarize(df: pd.DataFrame, keys: list[str], metrics: list[str], block: int, resamples: int,
              seed: int) -> pd.DataFrame:
    """Mean and 95% block-bootstrap interval of each metric per group; images in frame order."""
    rows = []
    for key, group in df.groupby(keys, sort=True):
        group = group.sort_values(["seq", "frame"])
        key = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            values = group[metric].to_numpy(np.float64)
            mean, lo, hi = block_bootstrap(values, block, resamples, seed)
            rows.append({**dict(zip(keys, key)), "metric": metric, "mean": mean, "lo": lo, "hi": hi,
                         "n_images": int(np.isfinite(values).sum())})
    return pd.DataFrame(rows)


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
                mean, lo, hi = block_bootstrap(diff, block, resamples, seed)
                rows.append({"model": model, "reference": reference, **dict(zip(keys, key)),
                             "metric": metric, "mean": mean, "lo": lo, "hi": hi,
                             "n_images": int(np.isfinite(diff).sum())})
    columns = ["model", "reference", *keys, "metric", "mean", "lo", "hi", "n_images"]
    return pd.DataFrame(rows, columns=columns)


def fmt(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "-"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
