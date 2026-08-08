# Development log

Working notes on decisions and bugs that aren't obvious from the code or
commit messages alone. Chronological. This is a scratch record for future
reference, not a polished changelog.

## Status

Phase 1 and Phase 2 are done (verified/corrected, see below). Phase 3's
first two boxes (train_quantile.py, conformal.py) are done. See the
project's own checklist for the authoritative phase-by-phase status.

Quantile models are now saved to disk: `models/q10.txt` ... `q90.txt`.

## Conformal: split vs adaptive on volatile vs stable days

No calendar events fall in the validation window (1886-1913), but there
is a real SNAP transition -- days 1890-1899 SNAP-active, surrounded by
inactive days -- used as the volatility signal instead. Both methods
compared at store-daily-aggregate granularity (summed P10/P50/P90 across
all 3,049 CA_1 items per day; sum-of-quantiles approximates the true
aggregate quantile, acceptable for this diagnostic, not a formal
guarantee). Split conformal calibrated once on the trailing 90 training
days (1796-1885, never touching the reserved test set 1914-1941 per
Section 15); adaptive conformal (ACI, gamma=0.1) walked forward through
the 28 validation days starting cold.

Raw coverage (split: 100%/83.3% volatile/stable; adaptive: 80%/77.8%)
is too noisy to read much into directly -- only 10 volatile and 18
stable days, so the standard error on these proportions is roughly
+-10-20 points.

**The robust finding is interval width**: adaptive conformal's band was
**24% wider on volatile days than stable days** (4068 vs 3284) purely
from reacting online to recent errors. Split conformal's correction is a
single static number from the calibration set -- it cannot adapt within
the validation window at all, regardless of which day it's applied to.
This is the actual theoretical distinction Section 6C asks to
demonstrate: adaptive responds to regime shifts; split only guarantees
*marginal*, not *conditional*, coverage. Raw output saved to
`models/conformal_volatile_vs_stable.csv`.

## Quantile models

`train_quantile_models()` trains 5 independent boosters (alpha 0.1, 0.25,
0.5, 0.75, 0.9), reusing the Optuna-tuned structural params (minus
`tweedie_variance_power`, irrelevant to the quantile objective). CA_1,
local: 1069s total (~18 min) for all 5, alpha=0.25 needed the most
rounds (432) before early stopping.

**Quantile crossing: 0.85% of adjacent-quantile pairs violate
monotonicity** (2,894 / 341,488 -- e.g. P25 < P10 for that row). Expected
and well-documented for independently-trained quantile regressors, not a
bug -- each booster has no knowledge of the others' predictions. Left
unaddressed in `train_quantile_models()` itself since conformal
calibration (next Phase 3 box) operates on top of these raw quantiles;
worth checking whether conformal calibration incidentally fixes this or
whether serving needs an explicit sort/clip step later.

## Design decisions worth remembering

**`build_features()` returns a pure feature matrix (X only), not X+y+id.**
The spec's own test (`test_feature_set_matches_spec`) requires
`set(build_features(df).columns)` to equal the feature-schema table
*exactly* — no label, no `id`/`date`/`d`. That's a deliberate X/y-style API:
the original row index is preserved (minus dropped null-price rows), so a
caller recovers `sales`/`id`/`date` via `original_df.loc[features.index, ...]`.
The Colab notebook's Cell 5 does exactly this before saving to parquet —
without it, the saved parquet would have no label and no way to identify
rows.

**Baseline evaluation is walk-forward, not single-origin.** The point/
quantile models use target-anchored direct forecasting (one origin, 28-day
horizon, no peeking at intermediate actuals). The three baselines
(`naive`, `seasonal_naive`, `moving_average`) are plain shift/rolling
functions per spec with no horizon parameter, so "3 baselines recorded"
was computed as a walk-forward evaluation instead: each validation day's
prediction uses whatever data precedes it, which can include earlier
validation-window actuals. This is a fair, standard way to benchmark
baselines, but it is *not* the same evaluation protocol the trained models
will get. Keep that in mind when comparing baseline RMSSE to model RMSSE
later — recompute baselines the same way if an apples-to-apples number is
needed.

**Baseline recording is CA_1-only, not the true 12-level WRMSSE.** Only the
CA_1 sample exists locally; Total/State-level aggregation needs all 10
stores, which only exists on Drive. The reported numbers are a single
pooled "CA_1 item-level" dollar-weighted RMSSE, not the full M5 metric.

## Bugs hit while actually running the notebook on real data

Every one of these passed the local synthetic tests and only broke against
real M5 data or a real Colab environment — exactly why Section 9 Cell 6
insists on a real-data check, not just the synthetic leakage tests.

1. **`d` column memory blowup.** `reshape_and_downcast()` downcast the six
   id columns and `sales`, but left `d` (e.g. `"d_1"`..`"d_1941"`, ~1,941
   unique values) as object dtype. Broadcast across 59.2M rows that's
   ~3.7GB of redundant string storage — the direct cause of the first
   Colab OOM crash. Fixed by also casting `d` to category.

2. **Calendar/price columns not downcast before merging.** `merge_sources()`
   merged in calendar's string columns (`weekday`, `event_name_1/2`,
   `event_type_1/2`) and int64 columns at their raw `read_csv` dtypes.
   Broadcast across 59.2M rows, the five string columns alone measured
   ~19GB as object dtype vs ~0.3GB as category — the second OOM crash,
   this time during the merge itself.

3. **`prices.store_id`/`item_id` still object dtype.** Fixed calendar and
   `wm_yr_wk`/`sell_price`, but missed that `prices`' own `store_id`/
   `item_id` were still object dtype, merging against `sales_long`'s
   already-categorical columns on the 59.2M-row side. A category-vs-object
   key mismatch on the large side can force pandas to expand the
   categorical back to object internally — same class of bug, third
   occurrence, third crash.

4. **Missing `observed=True` on categorical groupbys.** `id` keeps all
   30,490 category levels even after Cell 5 filters down to one store's
   ~3,049 series. Several `groupby(id_key, ...)` calls in
   `build_features()` didn't pass `observed=True`, so pandas iterated
   group machinery for ~27,000 empty categories per operation, on every
   per-store call — a real slowdown, not just warning noise.

5. **Cell 4/5 didn't create their target directories.** `to_parquet()`
   doesn't create nested directories. Cell 4 needed
   `os.makedirs('.../m5', exist_ok=True)`; Cell 5 needed a per-store
   `os.makedirs(store_dir, exist_ok=True)` before each write, not just the
   parent `full/` directory once.

6. **Cell 6's real-data leakage check sorted by the wrong column.**
   `.astype('category')` on `d` sorts categories lexicographically
   (`d_1, d_10, d_11, ..., d_2, d_20, ...`), not chronologically.
   `sort_values('d')` therefore scrambles row order. Cell 6 originally
   sorted by `d` before computing the expected `lag_28` shift, producing a
   false-positive mismatch (max diff 91.0) even though the actual stored
   `lag_28` values were correct (`build_features()` sorts by `date`
   internally, never by `d`). Fixed by sorting the verification by `date`
   instead. **Takeaway: never sort or filter chronologically by `d` — use
   `date`.** This applies to the Phase 2 train/validate/test split
   (days 1885/1886-1913/1914-1941) too.

7. **`rmsse()` overflowed on int16 sales.** Squaring a raw int16 array
   overflows for any day-to-day swing over ~181 units, which ordinary
   promotional spikes exceed, silently producing negative values under a
   square root -> NaN. Fixed by casting to float64 before any arithmetic
   in `rmsse()` and `pinball_loss()`.

8. **`event_cat` was object dtype, not category.** Built via string
   concatenation (`.astype(str) + "_" + ...`) with no final `.astype
   ("category")`. LightGBM auto-detects pandas `category` columns but
   errors on raw object columns -- would have failed at the first
   `train_tweedie()` call. Fixed in `pipeline.py`; `data/ca1_features
   .parquet` (generated before the fix) still has it as object, so
   `train_point.py`'s `_split_xy()` defensively casts any object column to
   category at the training boundary instead of requiring a Colab rerun.

9. **10-store combined dataset didn't fit in Colab RAM at full scale.**
   The 10 per-store parquets sum to ~11.6GB even after a float32 downcast
   on load; `pd.concat()` needs roughly 2x that at its peak (all source
   frames plus the new combined one), well past the ~12.7GB ceiling --
   crashed right after all 10 files finished loading, before printing the
   combined shape. Fixed by sampling 30% of item-store series per store
   (all stores represented, full day range per sampled item -- the
   lag/rolling features are already baked-in columns, so there's no
   continuity to preserve) for the global-vs-per-store comparison
   specifically. This is a real methodology choice, not just a memory
   workaround: the comparison answers "does global beat per-store?" on a
   representative subset, not literally every row.

10. **No training progress visibility during the slow Colab run.**
    `train_tweedie()` sets `"verbosity": -1` and the early-stopping
    callback's `verbose=False`, so Cell 4's ~40-minute run printed nothing
    until it finished. Checking Colab's RAM graph (steady ~6.8/12.7GB,
    not flatlined) was the only way to tell it was still actively
    computing rather than hung. Worth adding periodic eval logging
    (`lgb.log_evaluation(period=N)`) if a long unattended run needs to be
    monitored again.

## Baseline results (CA_1 sample, walk-forward, train<=1885 / validate 1886-1913)

| baseline | mean RMSSE | dollar-weighted RMSSE |
|---|---|---|
| naive | 0.9259 | 1.0549 |
| seasonal_naive | 0.9365 | 1.0610 |
| moving_average (window=28) | 0.7014 | 0.8176 |

3,049 CA_1 items evaluated. `moving_average` is the strongest of the three
on this proxy metric.

## CORRECTION: wrmsse_level silently dropped ~60% of items (2026-08-03)

The numbers below (point model 0.3275, global 0.3293, per-store 0.3302,
Optuna best_value 0.3258) were all computed with a real bug in
`wrmsse_level()`: it pivots `y_train` across all series in the level, and
any series not listed since the table's earliest date gets leading NaN
(no row before its own listing date, not a zero). `rmsse()`'s denominator
(`np.diff` then `np.mean`) returns NaN for any array containing NaN, and
`pandas.Series.sum()`'s default `skipna=True` then silently dropped those
NaN-scored series from the weighted sum -- while the weights still summed
to 1.0 over the *full* item set. On CA_1, **1,845 of 3,049 items (60%)**
were silently excluded this way. Fixed by `.dropna()` on each y_train
column before calling `rmsse()`, so the denominator reflects each
series' own available history instead of the pivot table's padding.

This does NOT affect the walk-forward baseline numbers below (naive
1.0549, seasonal_naive 1.0610, moving_average 0.8176) -- that script
computed each item's RMSSE from its own series directly, never through a
multi-series pivot, so it never hit this NaN-padding issue.

**Corrected, single-origin, apples-to-apples (all four evaluated on
CA_1's full 3,049 items, all using item-store WRMSSE with the fix):**

| method | WRMSSE |
|---|---|
| naive | 1.1182 |
| seasonal_naive | 1.0717 |
| moving_average | 0.8352 |
| point model | 0.7931 |

Point model still wins, but by a much smaller margin than originally
reported -- barely beats moving_average (~5%), not a 4x improvement.

**Both reruns done (2026-08-03), both with the fixed metric:**

- **Global vs per-store, full 10-store Colab data**: global **0.8462**
  vs per-store **0.8487** -- global still wins, and by nearly the same
  *relative* margin as the buggy run (~0.3% both times). Makes sense:
  both structures were affected by the same systematic item-dropping, so
  their relative comparison was distorted less than the absolute numbers
  were. Conclusion (global wins) holds.
- **Optuna, 30 trials, learning_rate floor raised to 0.02** (the
  original 0.01 floor produced the slowest trials without meaningfully
  better scores -- confirmed by this rerun finishing in 126s/trial average
  vs the first run's 4.5min/trial): best value **0.7907** (trial 25),
  essentially matching the untuned point model's 0.7931 -- tuning found
  only a ~0.3% improvement over the defaults already used. Saved to
  `models/best_params.json`, `models/optuna_study_v2.pkl`,
  `models/optuna_trials_v2.csv`.

## Point model results (SUPERSEDED, see correction above)

**Point model vs seasonal_naive** (CA_1 only, local, train<=1885 /
validate 1886-1913, item-store WRMSSE): point model **0.3275** vs
seasonal_naive's 1.0610 -- a clear win.

**Global vs per-store** (full 10-store dataset, 30% item sample, Colab,
same split): global **0.3293** vs per-store **0.3302** -- global wins,
narrowly. `store_id` as a feature in one model edges out training 10
separate boosters. Matches the artifact naming in the feature schema
reference table (`point_tweedie_global.txt` as "winning structure"),
which anticipated this outcome.

## Optuna study (SUPERSEDED, see correction above -- objective used the buggy metric)

Ran locally on CA_1 (75-trial run planned, ~56s/trial in a 3-trial dry
run). Stopped early at **39/75 trials** -- actual pace was much slower
than the dry run (~4.5 min/trial average, some low-learning-rate trials
needing far more boosting rounds before early stopping), and results had
clearly plateaued: best value moved from 0.32594 (trial 10) to 0.32583
(trial 38) over 28 more trials, a 0.03% improvement. Diminishing returns
from TPE having already found the good region.

Since the run script only persisted results after all 75 trials
completed, stopping it early meant reconstructing best_params.json and
optuna_trials.csv by parsing the captured trial log directly rather than
from a live Study object -- worth remembering: any long Optuna run should
either use RDB storage (`optuna.create_study(storage=...)`) so state
survives an interrupt, or persist incrementally, not just at the end.

Saved to `models/best_params.json` and `models/optuna_trials.csv`:

```json
{
  "num_leaves": 190, "learning_rate": 0.0115, "min_data_in_leaf": 171,
  "tweedie_variance_power": 1.233, "feature_fraction": 0.800,
  "bagging_fraction": 0.925, "bagging_freq": 3
}
```
best_value (item-store WRMSSE): 0.3258 -- essentially matching the
untuned point model's 0.3275, confirming the untuned defaults used
earlier were already close to reasonable.

## Repo housekeeping

- `CLAUDE.md` is intentionally gitignored (not on GitHub) per an explicit
  request partway through Phase 1. It still exists locally and still
  drives the workflow; its checklist edits just aren't reflected in git
  history going forward.
- `data/ca1_features.parquet` (~138MB) is gitignored — over GitHub's
  100MB file limit. Regenerate it via `notebooks/01_build_features.ipynb`.
