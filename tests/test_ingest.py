import pandas as pd

from src.features.ingest import merge_sources, reshape_and_downcast

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _small_wide_sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["ITEM_1_CA_1_evaluation", "ITEM_2_CA_1_evaluation"],
            "item_id": ["ITEM_1", "ITEM_2"],
            "dept_id": ["DEPT_1", "DEPT_1"],
            "cat_id": ["CAT_1", "CAT_1"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
            "d_1": [0, 5],
            "d_2": [1, 6],
            "d_3": [2, 7],
        }
    )


def _small_calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2011-01-29", "2011-01-30", "2011-01-31"],
            "wm_yr_wk": [11101, 11101, 11101],
            "weekday": ["Saturday", "Sunday", "Monday"],
            "wday": [1, 2, 3],
            "month": [1, 1, 1],
            "year": [2011, 2011, 2011],
            "d": ["d_1", "d_2", "d_3"],
            "event_name_1": [None, "SuperBowl", None],
            "event_type_1": [None, "Sporting", None],
            "event_name_2": [None, None, None],
            "event_type_2": [None, None, None],
            "snap_CA": [0, 1, 0],
            "snap_TX": [0, 0, 0],
            "snap_WI": [0, 0, 0],
        }
    )


def _small_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1"],
            "item_id": ["ITEM_1", "ITEM_2"],
            "wm_yr_wk": [11101, 11101],
            "sell_price": [3.5, 7.25],
        }
    )


def test_reshape_melts_wide_to_long():
    wide = _small_wide_sales()
    n_items = len(wide)
    n_days = 3
    long = reshape_and_downcast(wide)
    assert len(long) == n_items * n_days


def test_downcast_dtypes():
    long = reshape_and_downcast(_small_wide_sales())
    assert long["sales"].dtype == "int16"
    for col in ID_COLS:
        assert isinstance(long[col].dtype, pd.CategoricalDtype)
    assert isinstance(long["d"].dtype, pd.CategoricalDtype)


def test_merge_sources_preserves_row_count():
    long = reshape_and_downcast(_small_wide_sales())
    merged = merge_sources(long, _small_calendar(), _small_prices())
    assert len(merged) == len(long)


def test_merge_sources_calendar_join_correct():
    long = reshape_and_downcast(_small_wide_sales())
    merged = merge_sources(long, _small_calendar(), _small_prices())
    d2_rows = merged[merged["d"] == "d_2"]
    assert (d2_rows["event_name_1"] == "SuperBowl").all()
    assert (d2_rows["event_type_1"] == "Sporting").all()
    d1_rows = merged[merged["d"] == "d_1"]
    assert d1_rows["event_name_1"].isna().all()
