"""Pydantic request/response models. See CLAUDE.md Section 9 and Section 10."""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    item_id: str
    start_date: str
    scenario: dict = Field(default_factory=dict)
    service_level: float = 0.50


class PredictResponse(BaseModel):
    prediction_id: str
    dates: list[str]
    p10: list[float]
    p25: list[float]
    p50: list[float]
    p75: list[float]
    p90: list[float]
    history: list[float]
    order_qty: float
    stockout_risk: float
    expected_cost: float
    top_drivers: list[str]
    explanation: str


class ExplainRequest(BaseModel):
    prediction_id: str
    question: str | None = None
    history: list[dict] = Field(default_factory=list)


class ExplainResponse(BaseModel):
    explanation: str
    grounded_on: list[str]


class AlertItem(BaseModel):
    id: str
    store_id: str
    p90_forecast: float
    inventory_on_hand: float
    shortfall: float
    severity: str
