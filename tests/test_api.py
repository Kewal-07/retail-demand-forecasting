import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def sample_item_and_date(client):
    demo_data = client.app.state.demo_data
    item_id = demo_data["id"].iloc[0]
    item_rows = demo_data[demo_data["id"] == item_id].sort_values("date")
    start_date = item_rows["date"].iloc[100].strftime("%Y-%m-%d")
    return item_id, start_date


def test_health_returns_model_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_version"]


def test_predict_rejects_fixed_feature_override(client, sample_item_and_date):
    item_id, start_date = sample_item_and_date
    r = client.post(
        "/predict",
        json={"item_id": item_id, "start_date": start_date, "scenario": {"lag_28": 999}},
    )
    assert r.status_code == 422


def test_predict_rejects_untrained_service_level(client, sample_item_and_date):
    item_id, start_date = sample_item_and_date
    r = client.post(
        "/predict",
        json={"item_id": item_id, "start_date": start_date, "service_level": 0.67},
    )
    assert r.status_code == 422


def test_predict_fails_startup_on_feature_mismatch(tmp_path, monkeypatch):
    from api import model_loader

    (tmp_path / "feature_list.json").write_text(json.dumps(["wrong_feature_a", "wrong_feature_b"]))
    monkeypatch.setattr(model_loader, "MODELS_DIR", tmp_path)

    with pytest.raises(RuntimeError):
        model_loader.load_models()


def test_explain_grounded_on_lists_sources(client, sample_item_and_date):
    item_id, start_date = sample_item_and_date
    predict_r = client.post("/predict", json={"item_id": item_id, "start_date": start_date})
    assert predict_r.status_code == 200
    prediction_id = predict_r.json()["prediction_id"]

    explain_r = client.post(
        "/explain", json={"prediction_id": prediction_id, "question": None, "history": []}
    )
    assert explain_r.status_code == 200
    grounded_on = explain_r.json()["grounded_on"]
    assert "forecast" in grounded_on
    assert "shap_top5" in grounded_on


def test_explain_uses_current_prediction_not_default(client, sample_item_and_date, monkeypatch):
    # force the deterministic fallback template regardless of whether a
    # real GROQ_API_KEY happens to be set in this environment -- this
    # test's assertions check the fallback's exact numeric formatting, so
    # it must not depend on external service availability to be reliable
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    item_id, start_date = sample_item_and_date

    default_r = client.post("/predict", json={"item_id": item_id, "start_date": start_date})
    default_p50_total = sum(default_r.json()["p50"])

    scenario_r = client.post(
        "/predict",
        json={
            "item_id": item_id,
            "start_date": start_date,
            "scenario": {"sell_price": 999.0},
        },
    )
    scenario_p50_total = sum(scenario_r.json()["p50"])
    scenario_prediction_id = scenario_r.json()["prediction_id"]

    # the scenario override must actually have changed the forecast, or
    # this test can't distinguish "uses current prediction" from "always
    # uses whatever the default was"
    assert scenario_p50_total != default_p50_total

    explain_r = client.post(
        "/explain",
        json={"prediction_id": scenario_prediction_id, "question": None, "history": []},
    )
    assert explain_r.status_code == 200
    explanation = explain_r.json()["explanation"]

    # the fallback template names the total forecast demand explicitly,
    # so the scenario-modified number must appear, not the default one
    assert f"{scenario_p50_total:.1f}" in explanation
    assert f"{default_p50_total:.1f}" not in explanation
