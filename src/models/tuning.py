"""Hyperparameter search. See CLAUDE.md Section 6B and Section 9.

Tunes on the validation window, optimizing the same item-store WRMSSE used
to compare global vs per-store, not raw tweedie loss. Optuna's own
per-trial logging is left on (unlike train_tweedie's silenced LightGBM
output) so a long study still shows visible progress, one line per trial.
"""
import numpy as np
import optuna
import pandas as pd

from src.models.train_point import _item_store_wrmsse, _split_xy, train_tweedie


def optuna_search(train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int = 75) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            # floor raised from 0.01: the first study showed sub-0.02
            # learning rates need far more boosting rounds before early
            # stopping, dominating wall-clock time without meaningfully
            # beating higher-LR trials (best trial there used lr=0.0115,
            # but several lr>0.15 trials scored within 1% of it in a
            # fraction of the time)
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.05, 1.9),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }
        booster = train_tweedie(train_df, val_df, params)
        X_val, _ = _split_xy(val_df)
        y_pred = np.clip(booster.predict(X_val, num_iteration=booster.best_iteration), 0, None)
        return _item_store_wrmsse(train_df, val_df, y_pred)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study
