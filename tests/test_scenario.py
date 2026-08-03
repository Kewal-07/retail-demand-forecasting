import pandas as pd
import pytest

from src.features.scenario import apply_scenario


def _base_row() -> pd.Series:
    return pd.Series(
        {
            "sell_price": 10.0,
            "price_change_1w": 0.05,
            "price_rel_mean": 1.1,
            "price_rank_dept": 0.5,
            "weeks_since_price_change": 3,
            "event_name_1": None,
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
            "event_cat": "nan_CAT_1",
            "days_to_next_event": 12,
            "days_since_last_event": 4,
            "cat_id": "CAT_1",
            "lag_28": 7.0,
            "item_id": "ITEM_1",
        }
    )


def test_override_sell_price_recomputes_derived():
    row = _base_row()
    out = apply_scenario(row, {"sell_price": 15.0})
    assert out["price_change_1w"] != row["price_change_1w"]
    assert out["price_rel_mean"] != row["price_rel_mean"]
    assert out["price_rank_dept"] != row["price_rank_dept"]
    assert out["weeks_since_price_change"] != row["weeks_since_price_change"]
    assert out["weeks_since_price_change"] == 0
    assert out["sell_price"] == 15.0


def test_override_event_recomputes_day_distances():
    row = _base_row()
    out = apply_scenario(row, {"event_name_1": "SuperBowl", "event_type_1": "Sporting"})
    assert out["days_to_next_event"] != row["days_to_next_event"]
    assert out["days_since_last_event"] != row["days_since_last_event"]
    assert out["event_cat"] == "Sporting_CAT_1"


def test_override_fixed_field_raises():
    row = _base_row()
    with pytest.raises(ValueError):
        apply_scenario(row, {"lag_28": 999.0})
