"""Stockout alerts. See CLAUDE.md Section 9, 10, 11.

Flags an item-store as at risk when its P90 forecast (the demand level
the newsvendor layer would stock to cover 90% of the time) exceeds
current inventory on hand. Severity is the shortfall as a fraction of
the P90 forecast -- how much of the "stock enough to cover demand 90%
of the time" level is actually missing.
"""
import pandas as pd

SEVERITY_BINS = [0, 0.25, 0.5, 1.0]
SEVERITY_LABELS = ["low", "medium", "high"]


def generate_stockout_alerts(
    p90_forecast: pd.DataFrame, inventory_position: pd.DataFrame
) -> list[dict]:
    merged = p90_forecast.merge(inventory_position, on=["id", "store_id"], how="inner")
    merged["shortfall"] = merged["p90_forecast"] - merged["inventory_on_hand"]

    at_risk = merged[merged["shortfall"] > 0].copy()
    at_risk["severity"] = pd.cut(
        at_risk["shortfall"] / at_risk["p90_forecast"],
        bins=SEVERITY_BINS,
        labels=SEVERITY_LABELS,
    )

    return [
        {
            "id": row["id"],
            "store_id": row["store_id"],
            "p90_forecast": float(row["p90_forecast"]),
            "inventory_on_hand": float(row["inventory_on_hand"]),
            "shortfall": float(row["shortfall"]),
            "severity": str(row["severity"]),
        }
        for _, row in at_risk.iterrows()
    ]
