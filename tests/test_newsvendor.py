import pytest

from src.decision.newsvendor import order_quantity, service_level_from_costs


def test_order_quantity_equals_correct_quantile():
    service_level = service_level_from_costs(holding_cost=1, stockout_cost=3)
    assert service_level == pytest.approx(0.75)

    quantile_forecasts = {0.10: 5.0, 0.25: 8.0, 0.50: 12.0, 0.75: 16.0, 0.90: 22.0}
    assert order_quantity(quantile_forecasts, service_level) == 16.0


def test_service_level_slider_rejects_untrained_level():
    quantile_forecasts = {0.10: 5.0, 0.25: 8.0, 0.50: 12.0, 0.75: 16.0, 0.90: 22.0}
    with pytest.raises(ValueError):
        order_quantity(quantile_forecasts, service_level=0.67)
