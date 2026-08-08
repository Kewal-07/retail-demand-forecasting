import lightgbm as lgb
import numpy as np
import pandas as pd

from src.explain.shap_cache import compute_shap_cached, compute_shap_live
from src.features.pipeline import FEATURE_COLUMNS, build_features

N_DAYS = 80
N_ITEMS = 4


def _make_frame() -> pd.DataFrame:
    frames = []
    dates = pd.date_range("2013-11-01", periods=N_DAYS, freq="D")
    rng = np.random.RandomState(42)
    for i in range(N_ITEMS):
        # price varies day to day and sales responds inversely to it, so
        # the model actually has a price-sales relationship to learn --
        # a constant-per-item price (as in an earlier draft) carries zero
        # information for tree splits regardless of what SHAP computes
        price_series = 4.5 + i + np.sin(np.arange(N_DAYS) / 6) * 1.5
        base_sales = np.clip(9 - price_series, 1, None)
        sales = rng.poisson(base_sales).astype("float64")
        df = pd.DataFrame(
            {
                "id": f"ITEM_{i}_TX_1_evaluation",
                "item_id": f"ITEM_{i}",
                "dept_id": "DEPT_1",
                "cat_id": "CAT_1",
                "store_id": "TX_1",
                "state_id": "TX",
                "d": [f"d_{j + 1}" for j in range(N_DAYS)],
                "sales": sales,
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
                "snap_CA": (np.arange(N_DAYS) % 3 == 0).astype(int),
                "snap_TX": (np.arange(N_DAYS) % 4 == 0).astype(int),
                "snap_WI": (np.arange(N_DAYS) % 5 == 0).astype(int),
                "sell_price": price_series,
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _train_tiny_model():
    raw = _make_frame()
    features = build_features(raw)
    sales = raw.loc[features.index, "sales"]

    X = features.copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category")

    train_set = lgb.Dataset(X, label=sales)
    model = lgb.train({"objective": "tweedie", "verbosity": -1, "seed": 42}, train_set, num_boost_round=10)
    demo_df = features.iloc[-5:]
    return model, demo_df


def test_cached_and_live_agree_on_default_scenario():
    model, demo_df = _train_tiny_model()

    cached = compute_shap_cached(model, demo_df)
    live = compute_shap_live(model, demo_df)

    pd.testing.assert_frame_equal(cached, live)


def test_live_shap_changes_with_scenario():
    model, demo_df = _train_tiny_model()
    cached = compute_shap_cached(model, demo_df)

    scenario_rows = demo_df.copy()
    scenario_rows["sell_price"] = scenario_rows["sell_price"] * 3

    live = compute_shap_live(model, scenario_rows)

    assert not np.allclose(cached[FEATURE_COLUMNS].to_numpy(), live[FEATURE_COLUMNS].to_numpy())
