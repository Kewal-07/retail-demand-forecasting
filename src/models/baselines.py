"""Forecast baselines. See CLAUDE.md Section 9 and Section 13.

Each baseline forecasts day t using only data strictly before t, matching
how naive/seasonal_naive/moving_average are actually usable as forecasts
rather than in-sample fits.
"""
import pandas as pd


def naive(series: pd.Series) -> pd.Series:
    return series.shift(1)


def seasonal_naive(series: pd.Series, period: int = 7) -> pd.Series:
    return series.shift(period)


def moving_average(series: pd.Series, window: int = 28) -> pd.Series:
    return series.shift(1).rolling(window).mean()
