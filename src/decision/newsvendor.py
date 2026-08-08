"""Newsvendor decision layer. See CLAUDE.md Section 6D and Section 9."""
import math

TRAINED_SERVICE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]


def service_level_from_costs(holding_cost: float, stockout_cost: float) -> float:
    return stockout_cost / (stockout_cost + holding_cost)


def order_quantity(quantile_forecasts: dict[float, float], service_level: float) -> float:
    matched = next(
        (
            level
            for level in TRAINED_SERVICE_LEVELS
            if math.isclose(level, service_level, abs_tol=1e-9)
        ),
        None,
    )
    if matched is None:
        raise ValueError(
            f"service_level {service_level} is not one of the trained levels: "
            f"{TRAINED_SERVICE_LEVELS}"
        )
    return quantile_forecasts[matched]
