import pandas as pd

from src.models.baselines import moving_average, naive, seasonal_naive


def test_naive_shifts_by_one():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    pd.testing.assert_series_equal(naive(s), s.shift(1))


def test_seasonal_naive_shifts_by_seven():
    s = pd.Series(range(20), dtype="float64")
    pd.testing.assert_series_equal(seasonal_naive(s), s.shift(7))


def test_moving_average_window_28_matches_hand_calculation():
    s = pd.Series(range(40), dtype="float64")
    result = moving_average(s)
    pd.testing.assert_series_equal(result, s.shift(1).rolling(28).mean())

    # hand-check one value: forecast for day 28 (0-indexed) is the mean of
    # days 0..27, i.e. mean(range(28)) == 13.5
    assert result.iloc[28] == 13.5
