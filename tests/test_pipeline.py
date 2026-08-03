import numpy as np
import pandas as pd

from src.features.pipeline import FEATURE_COLUMNS, build_features

N_DAYS = 60


def _make_frame(state_id: str = "TX", n_days: int = N_DAYS) -> pd.DataFrame:
    dates = pd.date_range("2013-11-01", periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "id": "ITEM_1_TX_1_evaluation",
            "item_id": "ITEM_1",
            "dept_id": "DEPT_1",
            "cat_id": "CAT_1",
            "store_id": "TX_1",
            "state_id": state_id,
            "d": [f"d_{i + 1}" for i in range(n_days)],
            "sales": np.arange(n_days, dtype="float64") % 5,
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
            "sell_price": 4.5,
        }
    )
    return df


def test_feature_set_matches_spec():
    features = build_features(_make_frame())
    assert set(features.columns) == set(FEATURE_COLUMNS)


def test_snap_active_reads_correct_state():
    df = _make_frame(state_id="TX")
    features = build_features(df)
    aligned = df.loc[features.index]
    assert (features["snap_active"] == aligned["snap_TX"]).all()
    assert not (features["snap_active"] == aligned["snap_CA"]).all()


def test_event_cat_is_crossed_correctly():
    df = _make_frame()
    df["event_type_1"] = df["event_type_1"].astype(object)
    df.loc[df.index[10], "event_type_1"] = "Sporting"
    features = build_features(df)
    row = features.loc[df.index[10]]
    assert row["event_cat"] == "Sporting_CAT_1"


def test_is_closed_flags_christmas():
    dates = pd.date_range("2013-12-01", periods=40, freq="D")
    df = _make_frame(n_days=40)
    df["date"] = dates
    df["wm_yr_wk"] = (dates.year * 100 + dates.isocalendar().week.values).astype(int)
    df["month"] = dates.month.values
    features = build_features(df)
    christmas_idx = df.index[dates == pd.Timestamp("2013-12-25")]
    other_idx = df.index[dates == pd.Timestamp("2013-12-24")]
    assert (features.loc[christmas_idx, "is_closed"] == 1).all()
    assert (features.loc[other_idx, "is_closed"] == 0).all()


def test_rows_before_first_listing_dropped():
    df = _make_frame()
    cutoff = df.index[20]
    df.loc[df.index < cutoff, "sell_price"] = np.nan
    features = build_features(df)
    assert not any(idx < cutoff for idx in features.index)
    assert len(features) == len(df) - 20
