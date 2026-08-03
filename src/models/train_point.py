"""Point model training. See CLAUDE.md Section 6B and Section 9.

train_df/val_df are expected in the format the Colab notebook saves:
build_features()'s FEATURE_COLUMNS plus id/d/date/sales reattached by
index (see notebooks/01_build_features.ipynb Cell 5).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.pipeline import FEATURE_COLUMNS
from src.models.metrics import wrmsse_level


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category")
        elif X[col].dtype == "float64":
            # ~20 of the 44 feature columns are float64 (shift() on int16
            # sales produces float64 to hold NaN, and rolling/expanding
            # stats follow suit). At the full 10-store scale that is
            # several GB of unnecessary precision against a ~12.7GB Colab
            # ceiling; float32 is more than enough for these features.
            X[col] = X[col].astype("float32")
    return X, df["sales"]


def train_tweedie(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict) -> lgb.Booster:
    X_train, y_train = _split_xy(train_df)
    X_val, y_val = _split_xy(val_df)

    # free_raw_data defaults to True: LightGBM discards the raw frame
    # after binning into its internal histogram representation, which
    # matters at the full 10-store scale. Predictions later use a fresh
    # _split_xy() call, not this Dataset object, so nothing depends on the
    # raw data being retained here.
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    full_params = {"objective": "tweedie", "verbosity": -1, "seed": 42, **params}
    booster = lgb.train(
        full_params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return booster


def _dollar_weights(train_df: pd.DataFrame, train_end_day: int) -> pd.Series:
    day_num = train_df["d"].astype(str).str.replace("d_", "", regex=False).astype(int)
    tail = train_df[(day_num > train_end_day - 28) & (day_num <= train_end_day)]
    dollar_sales = (tail["sales"] * tail["sell_price"]).groupby(tail["id"], observed=True).sum()
    return dollar_sales / dollar_sales.sum()


def _item_store_wrmsse(train_df: pd.DataFrame, val_df: pd.DataFrame, y_pred: np.ndarray) -> float:
    val = val_df.copy()
    val["y_pred"] = y_pred

    y_true_wide = val.pivot(index="date", columns="id", values="sales")
    y_pred_wide = val.pivot(index="date", columns="id", values="y_pred")
    y_train_wide = train_df.pivot(index="date", columns="id", values="sales")

    day_num = train_df["d"].astype(str).str.replace("d_", "", regex=False).astype(int)
    weights = _dollar_weights(train_df, int(day_num.max()))

    return wrmsse_level(y_true_wide, y_pred_wide, y_train_wide, weights)


def compare_global_vs_store(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict) -> dict:
    global_booster = train_tweedie(train_df, val_df, params)
    X_val, _ = _split_xy(val_df)
    global_pred = global_booster.predict(X_val, num_iteration=global_booster.best_iteration)
    global_wrmsse = _item_store_wrmsse(train_df, val_df, global_pred)

    store_pred = pd.Series(index=val_df.index, dtype="float64")
    for store in train_df["store_id"].unique():
        train_slice = train_df[train_df["store_id"] == store]
        val_slice = val_df[val_df["store_id"] == store]
        if len(train_slice) == 0 or len(val_slice) == 0:
            continue
        booster = train_tweedie(train_slice, val_slice, params)
        X_val_slice, _ = _split_xy(val_slice)
        store_pred.loc[val_slice.index] = booster.predict(
            X_val_slice, num_iteration=booster.best_iteration
        )
    per_store_wrmsse = _item_store_wrmsse(train_df, val_df, store_pred.to_numpy())

    winner = "global" if global_wrmsse <= per_store_wrmsse else "per_store"
    return {
        "global_wrmsse": global_wrmsse,
        "per_store_wrmsse": per_store_wrmsse,
        "winner": winner,
    }
