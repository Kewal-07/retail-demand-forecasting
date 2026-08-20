import pandas as pd

from src.explain.retrieval import retrieve_calendar_window, retrieve_eda_passages


def test_calendar_window_is_inclusive_and_bounded():
    dates = pd.date_range("2016-01-01", periods=40, freq="D")
    calendar = pd.DataFrame({"date": dates, "event": [f"e{i}" for i in range(40)]})

    result = retrieve_calendar_window(calendar, start_date="2016-01-10", horizon=7)

    expected_dates = list(pd.date_range("2016-01-10", periods=7, freq="D"))
    assert list(result["date"]) == expected_dates


def test_eda_retrieval_returns_k_passages():
    eda_text = (
        "Sales show strong weekly seasonality with weekend peaks.\n\n"
        "SNAP benefit timing shifts purchasing patterns for eligible items.\n\n"
        "Price elasticity varies significantly across product categories.\n\n"
        "Christmas is the only day all stores close."
    )
    result = retrieve_eda_passages("weekly seasonality patterns", eda_text, k=2)

    assert len(result) == 2
    assert "seasonality" in result[0]
