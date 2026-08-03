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


def merge_sources(
    sales_long: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    merged = sales_long.merge(calendar, on="d", how="left")
    merged = merged.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    return merged
