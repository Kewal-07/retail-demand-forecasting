"""Quantile model training. See CLAUDE.md Section 6C and Section 9."""
import lightgbm as lgb
import pandas as pd

from src.models.train_point import _split_xy


def train_quantile_models(
    train_df: pd.DataFrame, val_df: pd.DataFrame, alphas: list[float], params: dict
) -> dict[float, lgb.Booster]:
    X_train, y_train = _split_xy(train_df)
    X_val, y_val = _split_xy(val_df)

    boosters = {}
    for alpha in alphas:
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        full_params = {
            "objective": "quantile",
            "alpha": alpha,
            "verbosity": -1,
            "seed": 42,
            **params,
        }
        boosters[alpha] = lgb.train(
            full_params,
            train_set,
            num_boost_round=2000,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
    return boosters
