"""Conformal calibration. See CLAUDE.md Section 6C and Section 9.

Two different objects are calibrated here and must not be confused: a
two-sided INTERVAL (calibrate_interval, for the chart band) and a
one-sided QUANTILE (calibrate_one_sided, for the newsvendor layer). An
80% two-sided interval and a 90% one-sided quantile are not the same
calibration and one is never substituted for the other.
"""
import numpy as np


def _finite_sample_quantile(scores: np.ndarray, level: float) -> float:
    n = len(scores)
    q_level = min(np.ceil((n + 1) * level) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def calibrate_interval(
    lower_preds: np.ndarray, upper_preds: np.ndarray, y_val: np.ndarray, alpha: float
) -> dict:
    # canonical conformalized quantile regression (Romano, Patterson &
    # Candes 2019): one combined nonconformity score per point (the worse
    # of the two one-sided violations), one q_hat at level (1 - alpha)
    # widens both sides -- a standard, exact finite-sample (1 - alpha)
    # marginal coverage guarantee under exchangeability.
    lower_preds = np.asarray(lower_preds)
    upper_preds = np.asarray(upper_preds)
    y_val = np.asarray(y_val)
    scores = np.maximum(lower_preds - y_val, y_val - upper_preds)
    q_hat = _finite_sample_quantile(scores, 1 - alpha)
    return {"q_hat_lower": q_hat, "q_hat_upper": q_hat}


def calibrate_one_sided(
    quantile_preds: np.ndarray, y_val: np.ndarray, service_level: float
) -> dict:
    scores = np.asarray(y_val) - np.asarray(quantile_preds)
    return {"q_hat": _finite_sample_quantile(scores, service_level)}


def adaptive_conformal_calibrate(
    quantile_preds: np.ndarray, y_test: np.ndarray, alpha: float, gamma: float
) -> dict:
    quantile_preds = np.asarray(quantile_preds, dtype="float64")
    y_test = np.asarray(y_test, dtype="float64")
    n = len(y_test)

    abs_errors = np.abs(y_test - quantile_preds)
    seen_errors: list[float] = []
    alpha_t = alpha
    intervals = np.zeros((n, 2))
    error_history = np.zeros(n, dtype=int)

    for t in range(n):
        if seen_errors:
            q_level = min(max(1 - alpha_t, 0.0), 1.0)
            q_t = float(np.quantile(seen_errors, q_level))
        else:
            q_t = 0.0

        lower, upper = quantile_preds[t] - q_t, quantile_preds[t] + q_t
        intervals[t] = (lower, upper)

        covered = lower <= y_test[t] <= upper
        err_t = 0 if covered else 1
        error_history[t] = err_t

        alpha_t = alpha_t + gamma * (alpha - err_t)
        seen_errors.append(abs_errors[t])

    return {"intervals": intervals, "error_history": error_history}
