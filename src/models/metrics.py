"""Evaluation metrics. See CLAUDE.md Section 9 and Section 13 for the exact
WRMSSE two-stage aggregation: dollar-weighted within a level, then an
unweighted mean across the 12 levels.
"""
import numpy as np
import pandas as pd


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    # cast first: sales arrives as int16 from the ingest pipeline, and
    # squaring a raw int16 diff overflows for any day-to-day swing over
    # ~181 units, which routine promotional spikes exceed
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    y_train = np.asarray(y_train, dtype="float64")
    numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
    naive_errors = np.diff(y_train)
    denominator = np.sqrt(np.mean(naive_errors**2))
    return float(numerator / denominator)


def series_weights(train_tail: pd.DataFrame, prices: pd.DataFrame, level: str) -> pd.Series:
    merged = train_tail.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    merged["dollar_sales"] = merged["sales"] * merged["sell_price"]
    group_key = pd.Series("Total", index=merged.index) if level == "Total" else merged[level]
    totals = merged.groupby(group_key, observed=True)["dollar_sales"].sum()
    return totals / totals.sum()


def wrmsse_level(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, y_train: pd.DataFrame, weights: pd.Series
) -> float:
    # y_train columns can carry leading NaN when built via pivot() across
    # series with different listing dates (a series not yet listed on the
    # table's earliest date has no row there, hence NaN, not zero). NaN
    # anywhere in y_train makes rmsse()'s denominator NaN, which pandas'
    # skipna=True default then silently drops from the weighted sum below
    # -- so any level with mixed listing dates would silently score only
    # the subset of series listed since day 1. dropna() first so the
    # denominator reflects each series' own available history instead.
    scores = pd.Series(
        {
            col: rmsse(
                y_true[col].to_numpy(),
                y_pred[col].to_numpy(),
                y_train[col].dropna().to_numpy(),
            )
            for col in y_true.columns
        }
    )
    aligned_weights = weights.reindex(scores.index)
    return float((scores * aligned_weights).sum())


def wrmsse(level_scores: dict[str, float]) -> float:
    return float(np.mean(list(level_scores.values())))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64")
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    contained = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(contained))
