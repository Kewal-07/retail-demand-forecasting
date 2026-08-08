import numpy as np

from src.models.conformal import (
    adaptive_conformal_calibrate,
    calibrate_interval,
    calibrate_one_sided,
)


def test_split_conformal_achieves_target_coverage_on_synthetic():
    rng = np.random.RandomState(42)
    n = 2000
    mu = rng.uniform(-5, 5, n)
    lower_preds = mu - 1.0
    upper_preds = mu + 1.0
    # noise wider than the +-1 band, so real miscoverage exists pre-calibration
    y = mu + rng.uniform(-2.5, 2.5, n)

    calib = slice(0, n // 2)
    test = slice(n // 2, n)

    alpha = 0.2  # target 80% coverage
    result = calibrate_interval(lower_preds[calib], upper_preds[calib], y[calib], alpha)

    calibrated_lower = lower_preds[test] - result["q_hat_lower"]
    calibrated_upper = upper_preds[test] + result["q_hat_upper"]
    coverage = np.mean((y[test] >= calibrated_lower) & (y[test] <= calibrated_upper))

    assert coverage >= (1 - alpha) - 0.03


def test_adaptive_conformal_widens_after_error_burst():
    rng = np.random.RandomState(42)
    n = 100
    quantile_preds = np.zeros(n)
    y_test = rng.normal(0, 0.1, n)

    burst_start, burst_end = 40, 50
    y_test[burst_start:burst_end] = rng.normal(0, 5.0, burst_end - burst_start)

    result = adaptive_conformal_calibrate(quantile_preds, y_test, alpha=0.1, gamma=0.05)
    widths = result["intervals"][:, 1] - result["intervals"][:, 0]

    width_before_burst = widths[:burst_start].mean()
    width_after_burst = widths[burst_end : burst_end + 10].mean()

    assert width_after_burst > width_before_burst


def test_one_sided_and_interval_calibrations_differ():
    rng = np.random.RandomState(42)
    n = 200
    y_val = rng.normal(50, 5, n)
    p10_preds = y_val - 8 + rng.normal(0, 0.3, n)
    p90_preds = y_val + 8 + rng.normal(0, 0.3, n)

    one_sided = calibrate_one_sided(p90_preds, y_val, service_level=0.9)
    two_sided = calibrate_interval(p10_preds, p90_preds, y_val, alpha=0.2)

    assert one_sided["q_hat"] != two_sided["q_hat_upper"]


def test_service_level_calibration_independent():
    rng = np.random.RandomState(42)
    n = 200
    preds = rng.normal(50, 5, n) - 5
    y_val = rng.normal(50, 5, n)

    result_75 = calibrate_one_sided(preds, y_val, service_level=0.75)
    result_90 = calibrate_one_sided(preds, y_val, service_level=0.90)

    assert result_75["q_hat"] != result_90["q_hat"]
