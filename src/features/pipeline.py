"""Feature engineering entry point. See CLAUDE.md Section 8 for the exact
column spec and Section 9 for the function signature.

build_features() returns exactly the Section 8 column set: engineered
columns plus the fixed/settable passthrough columns already present on the
input frame. Row identifiers (id, d, date, wm_yr_wk, weekday, year) and the
target (sales) are intentionally not part of that set -- they are not
features. The original row index is preserved (only null-sell_price rows
are dropped), so a caller can recover the label and identifiers by
indexing the pre-build_features frame with the output's index.
"""
import numpy as np
import pandas as pd

LAG_DAYS = [28, 35, 42, 56, 364]
ROLL_WINDOWS = [7, 28, 56]
ROLL_SHIFT = 28

FIXED_FEATURES = {
    *(f"lag_{n}" for n in LAG_DAYS),
    *(f"roll_mean_{w}_{ROLL_SHIFT}" for w in ROLL_WINDOWS),
    *(f"roll_std_{w}_{ROLL_SHIFT}" for w in ROLL_WINDOWS),
    "wday",
    "month",
    "week_of_year",
    "day_of_month",
    "days_since_last_sale",
    "zero_share_28",
    "mean_nonzero_sales",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
}

SETTABLE_FEATURES = {
    "sell_price",
    "price_change_1w",
    "price_rel_mean",
    "price_rank_dept",
    "weeks_since_price_change",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "snap_active",
    "event_name_1",
    "event_name_2",
    "event_type_1",
    "event_type_2",
    "event_cat",
    "days_to_next_event",
    "days_since_last_event",
    "is_closed",
}

FEATURE_COLUMNS = sorted(FIXED_FEATURES | SETTABLE_FEATURES)


def _days_since_last_true(flags: np.ndarray) -> np.ndarray:
    idx = np.arange(len(flags), dtype="float64")
    marker = np.where(flags, idx, np.nan)
    last_true = pd.Series(marker).ffill().to_numpy()
    return idx - last_true


def _event_distances(out: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    has_event = out["event_name_1"].notna() | out["event_name_2"].notna()
    event_days = np.sort(out.loc[has_event, "date"].unique())
    dates = out["date"].to_numpy()

    if len(event_days) == 0:
        nan_series = pd.Series(np.nan, index=out.index)
        return nan_series, nan_series

    idx_next = np.searchsorted(event_days, dates, side="left")
    has_next = idx_next < len(event_days)
    next_ev = event_days[np.clip(idx_next, 0, len(event_days) - 1)]
    days_to_next = np.where(
        has_next, (next_ev - dates).astype("timedelta64[D]").astype(float), np.nan
    )

    idx_prev = idx_next - 1
    has_prev = idx_prev >= 0
    prev_ev = event_days[np.clip(idx_prev, 0, len(event_days) - 1)]
    days_since_last = np.where(
        has_prev, (dates - prev_ev).astype("timedelta64[D]").astype(float), np.nan
    )

    return (
        pd.Series(days_to_next, index=out.index),
        pd.Series(days_since_last, index=out.index),
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["id", "date"]).copy()
    out["date"] = pd.to_datetime(out["date"])
    id_key = out["id"]

    # lags, fixed, all >= the 28-day horizon floor
    for n in LAG_DAYS:
        out[f"lag_{n}"] = out.groupby(id_key, observed=True, sort=False)["sales"].shift(n)

    # rolling mean/std, computed on the lag-28 series so the window itself
    # never reaches past t - 28
    lag28_sales = out.groupby(id_key, observed=True, sort=False)["sales"].shift(ROLL_SHIFT)
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}_{ROLL_SHIFT}"] = lag28_sales.groupby(id_key, observed=True, sort=False).transform(
            lambda s, w=w: s.rolling(w).mean()
        )
        out[f"roll_std_{w}_{ROLL_SHIFT}"] = lag28_sales.groupby(id_key, observed=True, sort=False).transform(
            lambda s, w=w: s.rolling(w).std()
        )

    # calendar, known in advance regardless of sales history, fixed
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["day_of_month"] = out["date"].dt.day

    # snap, settable
    out["snap_active"] = np.select(
        [out["state_id"] == "CA", out["state_id"] == "TX", out["state_id"] == "WI"],
        [out["snap_CA"], out["snap_TX"], out["snap_WI"]],
        default=0,
    )

    # events, settable, calendar-known so not subject to the t-28 rule
    out["event_cat"] = out["event_type_1"].astype(str) + "_" + out["cat_id"].astype(str)
    out["days_to_next_event"], out["days_since_last_event"] = _event_distances(out)

    # Christmas closure, settable
    out["is_closed"] = ((out["date"].dt.month == 12) & (out["date"].dt.day == 25)).astype(int)

    # price, settable. Price is known/assumed at the target date itself
    # (that is what the scenario slider overrides), not lagged like sales.
    price_1w_ago = out.groupby(id_key, observed=True, sort=False)["sell_price"].shift(7)
    out["price_change_1w"] = (out["sell_price"] - price_1w_ago) / price_1w_ago
    series_mean_price = out.groupby(id_key, observed=True, sort=False)["sell_price"].transform(
        "mean"
    )
    out["price_rel_mean"] = out["sell_price"] / series_mean_price
    out["price_rank_dept"] = out.groupby(
        ["store_id", "dept_id", "wm_yr_wk"], observed=True
    )["sell_price"].rank(pct=True)
    price_changed = out["sell_price"] != out.groupby(id_key, observed=True, sort=False)[
        "sell_price"
    ].shift(1)
    days_since_price_change = price_changed.groupby(id_key, observed=True, sort=False).transform(
        lambda s: pd.Series(_days_since_last_true(s.to_numpy()), index=s.index)
    )
    out["weeks_since_price_change"] = days_since_price_change // 7

    # intermittency, fixed. Computed causally (using history up to and
    # including day s) then shifted by 28, so the value used for target t
    # only reflects sales known at t - 28.
    zero_ind = (out["sales"] == 0).astype(float)
    zshare_causal = zero_ind.groupby(id_key, observed=True, sort=False).transform(
        lambda s: s.rolling(28, min_periods=1).mean()
    )
    out["zero_share_28"] = zshare_causal.groupby(id_key, observed=True, sort=False).shift(ROLL_SHIFT)

    nonzero_sales = out["sales"].where(out["sales"] != 0)
    mean_nz_causal = nonzero_sales.groupby(id_key, observed=True, sort=False).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    out["mean_nonzero_sales"] = mean_nz_causal.groupby(id_key, observed=True, sort=False).shift(ROLL_SHIFT)

    days_since_sale_causal = out.groupby(id_key, observed=True, sort=False)["sales"].transform(
        lambda s: pd.Series(_days_since_last_true((s != 0).to_numpy()), index=s.index)
    )
    out["days_since_last_sale"] = days_since_sale_causal.groupby(id_key, observed=True, sort=False).shift(
        ROLL_SHIFT
    )

    out = out[out["sell_price"].notna()]
    out = out.sort_index()
    return out[FEATURE_COLUMNS]
