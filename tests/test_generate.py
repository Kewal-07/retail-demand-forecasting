import os
from unittest.mock import MagicMock, patch

import pandas as pd

from src.explain.generate import build_explanation_context, call_groq


def test_context_includes_all_five_sources():
    context = build_explanation_context(
        forecast={"p50": [10, 12]},
        shap_top5=["lag_28", "price_change_1w"],
        calendar_events=pd.DataFrame({"event": ["SuperBowl"]}),
        price_history=pd.DataFrame({"sell_price": [4.5, 4.6]}),
        eda_passages=["some EDA note"],
    )

    expected_keys = {"forecast", "shap_top5", "calendar_events", "price_history", "eda_passages"}
    assert expected_keys <= set(context.keys())


def test_falls_back_to_template_when_groq_unavailable():
    context = {
        "shap_top5": ["lag_28", "price_change_1w", "snap_active"],
        "forecast": {"p50": [10, 12, 9]},
        "calendar_events": [],
        "price_history": [],
        "eda_passages": [],
    }

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-test"}):
        with patch("src.explain.generate.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("rate limited")
            mock_groq_cls.return_value = mock_client

            result = call_groq(context, question=None, history=[])

    assert result
    assert "lag_28" in result
