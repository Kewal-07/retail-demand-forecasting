import math

import numpy as np
import pandas as pd
import pytest

from src.models.metrics import (
    empirical_coverage,
    pinball_loss,
    rmsse,
    series_weights,
    wrmsse,
    wrmsse_level,
)


def test_rmsse_matches_hand_calculation():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # one-step errors all 1, denom = 1
    y_true = np.array([6.0, 7.0])
    y_pred = np.array([5.0, 5.0])  # squared errors 1, 4 -> mean 2.5
    expected = math.sqrt(2.5) / 1.0
    assert rmsse(y_true, y_pred, y_train) == pytest.approx(expected)


def test_series_weights_sum_to_one_per_level():
    train_tail = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1", "TX_1", "TX_1"],
            "item_id": ["I1", "I2", "I1", "I2"],
            "wm_yr_wk": [11101] * 4,
            "sales": [10, 5, 3, 2],
        }
    )
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1", "TX_1", "TX_1"],
            "item_id": ["I1", "I2", "I1", "I2"],
            "wm_yr_wk": [11101] * 4,
            "sell_price": [2.0, 3.0, 2.0, 3.0],
        }
    )
    for level in ["store_id", "item_id", "Total"]:
        weights = series_weights(train_tail, prices, level=level)
        assert weights.sum() == pytest.approx(1.0)


def test_wrmsse_level_is_dollar_weighted():
    y_train = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_true = pd.DataFrame({"A": [6.0, 6.0], "B": [6.0, 6.0]})
    weights = pd.Series({"A": 0.9, "B": 0.1})

    y_pred_perfect = y_true.copy()
    baseline_score = wrmsse_level(y_true, y_pred_perfect, y_train, weights)
    assert baseline_score == pytest.approx(0.0)

    y_pred_err_high = y_pred_perfect.copy()
    y_pred_err_high["A"] = [5.0, 5.0]
    score_high = wrmsse_level(y_true, y_pred_err_high, y_train, weights)

    y_pred_err_low = y_pred_perfect.copy()
    y_pred_err_low["B"] = [5.0, 5.0]
    score_low = wrmsse_level(y_true, y_pred_err_low, y_train, weights)

    assert score_high > score_low


def test_wrmsse_across_levels_is_unweighted_mean():
    level_scores = {f"level_{i}": float(i) for i in range(1, 13)}
    result = wrmsse(level_scores)
    plain_mean = sum(level_scores.values()) / len(level_scores)
    assert result == pytest.approx(plain_mean)

    # not a series-count-weighted mean: a level with far more series must
    # not count any more than a level with one series
    lopsided_scores = {"Total": 1.0, "Item_x_Store": 2.0}
    assert wrmsse(lopsided_scores) == pytest.approx(1.5)


def test_pinball_loss_symmetric_at_median():
    y_true = np.array([1.0, 2.0, 5.0, 3.0])
    y_pred = np.array([2.0, 1.0, 3.0, 6.0])
    mae = np.mean(np.abs(y_true - y_pred))
    assert pinball_loss(y_true, y_pred, alpha=0.5) == pytest.approx(mae / 2)


def test_coverage_calculation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = np.array([0.0, 0.0, 0.0, 5.0, 5.0])
    upper = np.array([10.0, 10.0, 2.0, 10.0, 10.0])
    # contained: 1,2,5 in-band (3 of 5); 3 and 4 out of band -> 0.6
    assert empirical_coverage(y_true, lower, upper) == pytest.approx(0.6)
