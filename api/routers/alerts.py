"""GET /alerts. See CLAUDE.md Section 9/10.

Local-dev scope: there is no real inventory system behind this API.
inventory_on_hand is a simple demo proxy (a fixed number of days' worth
of recent average sales), not real stock data.
"""
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, Request

from api.schemas import AlertItem
from src.decision.alerts import generate_stockout_alerts
from src.features.pipeline import FEATURE_COLUMNS

router = APIRouter()

DEMO_INVENTORY_DAYS_COVER = 3


@router.get("/alerts", response_model=list[AlertItem])
def alerts(
    request: Request, store_id: str, limit: int = Query(default=20, le=100)
) -> list[AlertItem]:
    demo_data: pd.DataFrame = request.app.state.demo_data
    models = request.app.state.models

    store_rows = demo_data[demo_data["store_id"] == store_id]
    if store_rows.empty:
        return []

    latest_day = store_rows["day_num"].max()
    latest_rows = store_rows[store_rows["day_num"] == latest_day].copy()

    X = latest_rows[FEATURE_COLUMNS].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category")
    p90_pred = np.clip(models["q90"].predict(X), 0, None)

    recent = store_rows[store_rows["day_num"] > latest_day - 28]
    avg_sales = recent.groupby("id", observed=True)["sales"].mean()

    p90_forecast = pd.DataFrame(
        {"id": latest_rows["id"].to_numpy(), "store_id": store_id, "p90_forecast": p90_pred}
    )
    inventory_position = pd.DataFrame(
        {
            "id": avg_sales.index,
            "store_id": store_id,
            "inventory_on_hand": (avg_sales * DEMO_INVENTORY_DAYS_COVER).to_numpy(),
        }
    )

    alert_dicts = generate_stockout_alerts(p90_forecast, inventory_position)
    alert_dicts.sort(key=lambda a: a["shortfall"], reverse=True)
    return [AlertItem(**a) for a in alert_dicts[:limit]]
