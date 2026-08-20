"""FastAPI app. See CLAUDE.md Section 9/10/15.

Models load once at import time (module scope), so a feature_list.json
mismatch fails at startup, not on the first request (Section 15: "MUST
validate feature order at API startup").

Local-dev scope note: predict/explain/alerts serve from the CA_1 sample
(data/ca1_features.parquet) as the "database", matching the project's
"iterate on CA_1 locally" approach -- there is no live feature pipeline
or real inventory system behind this API yet, both are out of scope for
Phase 5.
"""
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI

from api.model_loader import load_models
from api.routers import alerts, explain, health, predict
from src.features.pipeline import FEATURE_COLUMNS

load_dotenv()  # GROQ_API_KEY, read by src/explain/generate.py's call_groq()

_START_TIME = time.time()

MODELS = load_models()  # fails at import/startup on a feature order mismatch

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ca1_features.parquet"
DEMO_DATA = pd.read_parquet(DATA_PATH)
DEMO_DATA["day_num"] = (
    DEMO_DATA["d"].astype(str).str.replace("d_", "", regex=False).astype(int)
)
# establish a canonical category dtype for any feature column that came
# out of the parquet as object (e.g. event_cat, saved before pipeline.py
# was fixed to emit it as category) ONCE here, at the full-dataset scope
# -- so any window sliced from DEMO_DATA later already carries the
# correct categorical structure, matching what the boosters were trained
# with, rather than each request re-deriving categories from just its
# own 28-row window
for col in FEATURE_COLUMNS:
    if DEMO_DATA[col].dtype == object:
        DEMO_DATA[col] = DEMO_DATA[col].astype("category")

app = FastAPI(title="Stockwise")
app.state.models = MODELS
app.state.demo_data = DEMO_DATA
app.state.start_time = _START_TIME
app.state.prediction_store = {}

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(alerts.router)
