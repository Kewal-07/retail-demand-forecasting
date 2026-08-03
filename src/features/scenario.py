"""Single-row scenario overrides for live serving. See CLAUDE.md Section 9.

apply_scenario() operates on one already-featured row (as produced by
build_features(), one row per forecast target date) rather than a whole
frame, since it is called from the live /predict path where only the row
being served is available. Overriding sell_price or an event field
recomputes exactly the derived columns Section 8 lists as depending on it;
overriding a fixed column is rejected.
"""
import numpy as np
import pandas as pd

from src.features.pipeline import FIXED_FEATURES, SETTABLE_FEATURES

PRICE_DERIVED_FIELDS = {
    "price_change_1w",
    "price_rel_mean",
    "price_rank_dept",
    "weeks_since_price_change",
}
EVENT_FIELDS = {"event_name_1", "event_name_2", "event_type_1", "event_type_2"}


def apply_scenario(row: pd.Series, scenario: dict) -> pd.Series:
    for key in scenario:
        if key in FIXED_FEATURES:
            raise ValueError(f"{key} is a fixed feature and cannot be overridden by a scenario")
        if key not in SETTABLE_FEATURES:
            raise ValueError(f"{key} is not a recognized settable feature")

    out = row.copy()
    for key, value in scenario.items():
        out[key] = value

    if "sell_price" in scenario:
        old_price = row.get("sell_price")
        new_price = scenario["sell_price"]
        if pd.notna(old_price) and old_price != 0:
            old_price_change_1w = row.get("price_change_1w")
            if pd.notna(old_price_change_1w) and (1 + old_price_change_1w) != 0:
                price_1w_ago = old_price / (1 + old_price_change_1w)
                if price_1w_ago != 0:
                    out["price_change_1w"] = (new_price - price_1w_ago) / price_1w_ago

            old_price_rel_mean = row.get("price_rel_mean")
            if pd.notna(old_price_rel_mean) and old_price_rel_mean != 0:
                series_mean_price = old_price / old_price_rel_mean
                out["price_rel_mean"] = new_price / series_mean_price

            old_price_rank_dept = row.get("price_rank_dept")
            if pd.notna(old_price_rank_dept):
                out["price_rank_dept"] = float(
                    np.clip(old_price_rank_dept * (new_price / old_price), 0, 1)
                )

        out["weeks_since_price_change"] = 0 if new_price != old_price else row.get(
            "weeks_since_price_change"
        )

    if EVENT_FIELDS & scenario.keys():
        if "event_type_1" in scenario:
            out["event_cat"] = f"{scenario['event_type_1']}_{row.get('cat_id')}"
        has_event = pd.notna(out.get("event_name_1")) or pd.notna(out.get("event_name_2"))
        if has_event:
            out["days_to_next_event"] = 0
            out["days_since_last_event"] = 0

    return out
