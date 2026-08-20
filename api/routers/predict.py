"""POST /predict. See CLAUDE.md Section 9/10.

Local-dev scope: serves from the CA_1 sample loaded into app.state at
startup (see api/main.py), not a live feature pipeline. Horizon is
always 28 days -- not a request parameter, per Section 6A.
"""
import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.schemas import PredictRequest, PredictResponse
from src.decision.newsvendor import TRAINED_SERVICE_LEVELS
from src.explain.generate import build_explanation_context, call_groq
from src.explain.retrieval import retrieve_calendar_window
from src.explain.shap_cache import compute_shap_live
from src.features.pipeline import FEATURE_COLUMNS
from src.features.scenario import apply_scenario

router = APIRouter()

HORIZON = 28
HOLDING_COST = 1.0
QUANTILE_ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]


def _quantile_model_name(service_level: float) -> str:
    return f"q{int(round(service_level * 100)):02d}"


def _predict_quantiles(models: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        _quantile_model_name(alpha): np.clip(models[_quantile_model_name(alpha)].predict(X), 0, None)
        for alpha in QUANTILE_ALPHAS
    }


def _summarize_explain_context(
    item_rows: pd.DataFrame, start_date: str, start: pd.Timestamp
) -> tuple[list[str], str]:
    # Dumping 28 raw daily rows (mostly NaN events, near-identical prices)
    # straight into the LLM prompt produced an extremely long, repetitive
    # block that caused the model to degenerate into blank output in
    # testing -- summarize to only what's actually informative instead.
    window_calendar = retrieve_calendar_window(
        item_rows[["date", "event_name_1", "event_name_2"]], start_date, HORIZON
    )
    events = window_calendar[
        window_calendar["event_name_1"].notna() | window_calendar["event_name_2"].notna()
    ]
    calendar_events = [
        f"{row['date'].date()}: {row['event_name_1'] or row['event_name_2']}"
        for _, row in events.iterrows()
    ]

    recent_prices = item_rows[item_rows["date"] < start][["date", "sell_price"]].tail(HORIZON)
    if recent_prices.empty:
        price_history = "no prior price history available"
    else:
        current_price = recent_prices["sell_price"].iloc[-1]
        earliest_price = recent_prices["sell_price"].iloc[0]
        if current_price == earliest_price:
            price_history = f"steady at ${current_price:.2f} over the prior {len(recent_prices)} days"
        else:
            price_history = (
                f"changed from ${earliest_price:.2f} to ${current_price:.2f} "
                f"over the prior {len(recent_prices)} days"
            )

    return calendar_events, price_history


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    if not any(abs(body.service_level - lvl) < 1e-9 for lvl in TRAINED_SERVICE_LEVELS):
        raise HTTPException(
            status_code=422,
            detail=(
                f"service_level {body.service_level} is not one of the trained "
                f"levels: {TRAINED_SERVICE_LEVELS}"
            ),
        )

    demo_data: pd.DataFrame = request.app.state.demo_data
    item_rows = demo_data[demo_data["id"] == body.item_id].sort_values("date")
    if item_rows.empty:
        raise HTTPException(status_code=422, detail=f"item_id {body.item_id!r} not found")

    start = pd.to_datetime(body.start_date)
    end = start + pd.Timedelta(days=HORIZON - 1)
    window = item_rows[(item_rows["date"] >= start) & (item_rows["date"] <= end)].sort_values("date")

    if len(window) != HORIZON:
        valid_min = item_rows["date"].min()
        valid_max = item_rows["date"].max() - pd.Timedelta(days=HORIZON - 1)
        raise HTTPException(
            status_code=422,
            detail=(
                f"start_date {body.start_date} is outside the available data range; "
                f"valid range is {valid_min.date()} to {valid_max.date()}"
            ),
        )

    try:
        if body.scenario:
            original_dtypes = window.dtypes
            window = pd.DataFrame(
                [apply_scenario(row, body.scenario) for _, row in window.iterrows()]
            )
            # apply_scenario() returns plain per-row Series, so rebuilding a
            # DataFrame from a list of them loses category dtype entirely --
            # restore each column's original categories explicitly, rather
            # than re-deriving a category dtype from this 28-row window
            # alone, which would use a different category encoding than
            # what the boosters were trained with and fail at predict time.
            for col, dtype in original_dtypes.items():
                if isinstance(dtype, pd.CategoricalDtype):
                    window[col] = window[col].astype(dtype)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    models = request.app.state.models
    X = window[FEATURE_COLUMNS].copy()

    quantile_preds = _predict_quantiles(models, X)

    history_window = item_rows[item_rows["date"] < start].tail(HORIZON)
    history = [float(v) for v in history_window["sales"]]

    service_level_col = _quantile_model_name(body.service_level)
    order_qty = float(quantile_preds[service_level_col].sum())
    stockout_risk = 1.0 - body.service_level

    p50_sum = float(quantile_preds["q50"].sum())
    stockout_cost = (
        HOLDING_COST * body.service_level / (1 - body.service_level)
        if body.service_level < 1
        else float("inf")
    )
    expected_cost = HOLDING_COST * max(order_qty - p50_sum, 0) + stockout_cost * max(
        p50_sum - order_qty, 0
    )

    shap_values = compute_shap_live(models["point"], window)
    top_drivers = shap_values.abs().mean().sort_values(ascending=False).head(5).index.tolist()

    calendar_events, price_history = _summarize_explain_context(item_rows, body.start_date, start)

    context = build_explanation_context(
        forecast={"p50": quantile_preds["q50"].tolist()},
        shap_top5=top_drivers,
        calendar_events=calendar_events,
        price_history=price_history,
        eda_passages=[],
    )
    explanation = call_groq(context, question=None, history=[])

    prediction_id = str(uuid.uuid4())
    response = PredictResponse(
        prediction_id=prediction_id,
        dates=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in window["date"]],
        p10=quantile_preds["q10"].tolist(),
        p25=quantile_preds["q25"].tolist(),
        p50=quantile_preds["q50"].tolist(),
        p75=quantile_preds["q75"].tolist(),
        p90=quantile_preds["q90"].tolist(),
        history=history,
        order_qty=order_qty,
        stockout_risk=stockout_risk,
        expected_cost=expected_cost,
        top_drivers=top_drivers,
        explanation=explanation,
    )

    request.app.state.prediction_store[prediction_id] = {"response": response, "context": context}
    return response
