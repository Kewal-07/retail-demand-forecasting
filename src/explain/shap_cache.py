"""SHAP explanation attribution. See CLAUDE.md Section 6C and Section 9.

compute_shap_cached and compute_shap_live are the same underlying
TreeExplainer computation -- the split between them is about *when* each
gets called (cached once for the default demo view; live on-demand for a
28-row scenario-modified batch), not a difference in algorithm. Cached
values go stale the instant a scenario changes, which is the whole reason
the live path exists.
"""
import lightgbm as lgb
import pandas as pd
import shap

from src.features.pipeline import FEATURE_COLUMNS


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category")
    return X


def _shap_values(model: lgb.Booster, df: pd.DataFrame) -> pd.DataFrame:
    X = _prepare_X(df)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    return pd.DataFrame(values, columns=FEATURE_COLUMNS, index=df.index)


def compute_shap_cached(model: lgb.Booster, demo_df: pd.DataFrame) -> pd.DataFrame:
    return _shap_values(model, demo_df)


def compute_shap_live(model: lgb.Booster, scenario_rows: pd.DataFrame) -> pd.DataFrame:
    return _shap_values(model, scenario_rows)
