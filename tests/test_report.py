import numpy as np
import pandas as pd
import pytest

from src.report import fmt, paired, summarize


def per_image(model, values, subset="all"):
    return pd.DataFrame({
        "seq": "00001", "frame": np.arange(len(values)) * 5, "model": model, "subset": subset,
        "absrel": values,
    })


def test_summarize_means_and_counts_non_nan_images():
    df = per_image("a", [0.1, 0.2, np.nan, 0.3] * 10)
    out = summarize(df, ["model", "subset"], ["absrel"], block=4, resamples=200, seed=0)
    row = out.iloc[0]
    assert (row["model"], row["subset"], row["metric"]) == ("a", "all", "absrel")
    assert row["mean"] == pytest.approx(0.2) and row["n_images"] == 30
    assert row["lo"] <= row["mean"] <= row["hi"]


def test_summarize_orders_images_by_frame_before_blocking():
    shuffled = per_image("a", np.arange(40, dtype=float)).sample(frac=1, random_state=0)
    ordered = per_image("a", np.arange(40, dtype=float))
    a = summarize(shuffled, ["model"], ["absrel"], 10, 300, 0)
    b = summarize(ordered, ["model"], ["absrel"], 10, 300, 0)
    pd.testing.assert_frame_equal(a, b)


def test_paired_difference_against_reference_on_shared_images():
    ref = per_image("ref", [0.2] * 20)
    other = per_image("b", [0.3] * 20)
    other.loc[3, "absrel"] = np.nan
    out = paired(pd.concat([ref, other]), ["subset"], ["absrel"], "ref", 5, 100, 0)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["model"] == "b" and row["mean"] == pytest.approx(0.1) and row["n_images"] == 19


def test_fmt():
    assert fmt(0.12345, 0.1, 0.15) == "0.123 [0.100, 0.150]"
    assert fmt(np.nan, np.nan, np.nan) == "-"


def test_paired_with_only_the_reference_is_empty_but_has_columns():
    out = paired(per_image("ref", [0.2] * 5), ["subset"], ["absrel"], "ref", 5, 10, 0)
    assert out.empty and {"model", "subset", "mean"} <= set(out.columns)
