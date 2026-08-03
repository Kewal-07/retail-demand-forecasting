"""
Leakage tests for src/features/pipeline.py.

Written first, against an empty src/features/, per CLAUDE.md Section 12.
The core invariant checked everywhere in this file: no feature computed by
build_features() may read sales data newer than t - 28.
"""
import re

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import build_features

N_DAYS = 450


def _make_synthetic_frame(n_days: int = N_DAYS) -> pd.DataFrame:
    """One item-store series, in the post-merge_sources schema, sales[t] == t."""
    dates = pd.date_range("2013-01-01", periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "id": "ITEM_1_CA_1_evaluation",
            "item_id": "ITEM_1",
            "dept_id": "DEPT_1",
            "cat_id": "CAT_1",
            "store_id": "CA_1",
            "state_id": "CA",
            "d": [f"d_{i + 1}" for i in range(n_days)],
            "sales": np.arange(n_days, dtype="float64"),
            "date": dates,
            "wm_yr_wk": (dates.year * 100 + dates.isocalendar().week.values).astype(int),
            "weekday": dates.day_name(),
            "wday": dates.dayofweek.values + 1,
            "month": dates.month.values,
            "year": dates.year.values,
            "event_name_1": np.nan,
            "event_type_1": np.nan,
            "event_name_2": np.nan,
            "event_type_2": np.nan,
            "snap_CA": (np.arange(n_days) % 3 == 0).astype(int),
            "snap_TX": (np.arange(n_days) % 4 == 0).astype(int),
            "snap_WI": (np.arange(n_days) % 5 == 0).astype(int),
            "sell_price": 9.99,
        }
    )
    return df


LAG_RE = re.compile(r"^lag_(\d+)$")
ROLL_RE = re.compile(r"^roll_(mean|std)_(\d+)_(\d+)$")


def test_no_lag_shorter_than_horizon():
    features = build_features(_make_synthetic_frame())
    lag_cols = [c for c in features.columns if LAG_RE.match(c)]
    assert lag_cols, "expected at least one lag_N column"
    for col in lag_cols:
        n = int(LAG_RE.match(col).group(1))
        assert n >= 28, f"{col} has lag shorter than the 28-day horizon"
    assert "lag_7" not in features.columns
    assert "lag_14" not in features.columns


def test_lag_values_match_known_history():
    df = _make_synthetic_frame()
    features = build_features(df)
    t = N_DAYS - 1
    row = features.iloc[t]
    for n in (28, 35, 42, 56, 364):
        col = f"lag_{n}"
        assert row[col] == pytest.approx(t - n), f"{col} at row t={t} should equal t-{n}"


def test_rolling_windows_end_at_or_before_horizon():
    features = build_features(_make_synthetic_frame())
    roll_cols = [c for c in features.columns if ROLL_RE.match(c)]
    assert roll_cols, "expected at least one roll_{stat}_{window}_{shift} column"
    for col in roll_cols:
        _, _, shift = ROLL_RE.match(col).groups()
        assert int(shift) >= 28, f"{col} has a rolling shift shorter than the 28-day horizon"


def test_rolling_mean_values_match_known_history():
    df = _make_synthetic_frame()
    features = build_features(df)
    shifted = df["sales"].shift(28)
    for window in (7, 28, 56):
        expected = shifted.rolling(window).mean()
        col = f"roll_mean_{window}_28"
        assert col in features.columns, f"missing expected column {col}"
        pd.testing.assert_series_equal(
            features[col].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )


def test_future_sales_do_not_change_features():
    df = _make_synthetic_frame()
    baseline = build_features(df)

    t = N_DAYS - 50
    mutated = df.copy()
    cutoff = t - 28
    mutated.loc[mutated.index > cutoff, "sales"] = 999_999
    mutated_features = build_features(mutated)

    input_cols = set(df.columns)
    engineered_cols = [c for c in baseline.columns if c not in input_cols]
    assert engineered_cols, "expected build_features to add engineered columns"

    base_row = baseline.iloc[t]
    mut_row = mutated_features.iloc[t]
    for col in engineered_cols:
        b, m = base_row[col], mut_row[col]
        if pd.isna(b) and pd.isna(m):
            continue
        assert b == pytest.approx(m), (
            f"{col} at row t={t} changed after mutating sales newer than t-28 "
            f"(baseline={b!r}, mutated={m!r})"
        )
