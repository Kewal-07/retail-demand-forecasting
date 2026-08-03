"""Load and reshape the raw M5 CSVs. See CLAUDE.md Section 9."""
import os

import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_raw(data_dir: str) -> dict[str, pd.DataFrame]:
    return {
        "sales": pd.read_csv(os.path.join(data_dir, "sales_train_evaluation.csv")),
        "calendar": pd.read_csv(os.path.join(data_dir, "calendar.csv")),
        "prices": pd.read_csv(os.path.join(data_dir, "sell_prices.csv")),
    }


def reshape_and_downcast(sales_wide: pd.DataFrame) -> pd.DataFrame:
    day_cols = [c for c in sales_wide.columns if c not in ID_COLS]
    sales_long = sales_wide.melt(
        id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales"
    )
    sales_long["sales"] = sales_long["sales"].astype("int16")
    for col in ID_COLS:
        sales_long[col] = sales_long[col].astype("category")
    # d has ~1,941 unique values repeated across 59.2M rows; left as object
    # dtype this is several GB of redundant string storage.
    sales_long["d"] = sales_long["d"].astype("category")
    return sales_long


CALENDAR_CATEGORY_COLS = [
    "weekday",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]
CALENDAR_INT8_COLS = ["wday", "month", "snap_CA", "snap_TX", "snap_WI"]


def merge_sources(
    sales_long: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    # calendar and prices are tiny on their own, but every column here gets
    # broadcast across all 59.2M rows of sales_long by the merge below. Left
    # as the default read_csv dtypes (object strings, int64), that broadcast
    # is several GB per column; downcast first so the merge stays cheap.
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    for col in CALENDAR_CATEGORY_COLS:
        calendar[col] = calendar[col].astype("category")
    for col in CALENDAR_INT8_COLS:
        calendar[col] = calendar[col].astype("int8")
    calendar["year"] = calendar["year"].astype("int16")
    calendar["wm_yr_wk"] = calendar["wm_yr_wk"].astype("int32")

    prices = prices.copy()
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int32")
    prices["sell_price"] = prices["sell_price"].astype("float32")

    merged = sales_long.merge(calendar, on="d", how="left")
    merged = merged.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    return merged
