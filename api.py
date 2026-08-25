"""
Phase 2 - Step 3: The Prediction API
---------------------------------------
A small FastAPI service that sits between the Leaflet frontend (on
Vercel) and the forecasting logic from Step 1 + Step 2.

Endpoint:
    POST /predict
        body: { "cases": [{lat, lng, day}, ...],
                "target_day": int,
                "population_estimate": int (optional) }
        returns: predicted centroid, radius (trend-only and
                 SIR-scaled), predicted case count, and the fitted
                 SIR parameters -- everything the frontend needs to
                 draw the "Predicted N-Day Expansion" polygon.

Run locally:
    uvicorn phase2_api:app --reload --port 8000

Deployment note: Vercel's serverless functions don't run long-lived
Python processes well, so this needs a separate host for the API
itself (Render / Railway / Fly.io all have free tiers big enough for
a BTP demo). The frontend stays on Vercel and just fetch()es this
service's URL -- see the CORS config below, which you'll need to
update with your actual Vercel domain before deploying.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_engineering import compute_daily_features
from forecasting_engine import forecast_outbreak

app = FastAPI(title="Outbreak Prediction API", version="0.1.0")

# --- CORS ---------------------------------------------------------------
# Update allow_origins with your real Vercel deployment URL before
# shipping. "*" is fine for local dev only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://outbreak-spatial-mapper.vercel.app",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# --- Request / response schemas -----------------------------------------

class CaseIn(BaseModel):
    lat: float
    lng: float
    day: int = Field(..., ge=1)


class PredictRequest(BaseModel):
    cases: List[CaseIn]
    target_day: int = Field(..., ge=1)
    population_estimate: int = Field(3000, ge=1, description="SIR total population assumption")


class PredictResponse(BaseModel):
    target_day: int
    centroid_lat: float
    centroid_lng: float
    radius_km_trend_only: float
    radius_km_sir_scaled: float
    radius_uncertainty_km: float
    predicted_case_count: float
    sir_beta: float
    sir_gamma: float
    sir_population_assumed: int


# --- Routes ---------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.cases) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 logged cases to fit a forecast.")

    max_observed_day = max(c.day for c in req.cases)
    if req.target_day <= max_observed_day:
        raise HTTPException(
            status_code=400,
            detail=f"target_day ({req.target_day}) must be after the latest observed day ({max_observed_day}).",
        )

    df = pd.DataFrame(
        [{"Point_ID": i, "Latitude": c.lat, "Longitude": c.lng, "Timeline_Day": c.day}
         for i, c in enumerate(req.cases, start=1)]
    )

    daily = compute_daily_features(df, cumulative=True)

    # Need at least a couple of distinct days with growth for the SIR
    # fit to be meaningful, not just numerically defined.
    if daily["day"].nunique() < 3:
        raise HTTPException(
            status_code=400,
            detail="Need cases spread across at least 3 distinct days to fit a trend.",
        )

    try:
        result = forecast_outbreak(
            daily_features=daily,
            target_day=req.target_day,
            population_estimate=req.population_estimate,
        )
    except RuntimeError as e:
        # curve_fit raises RuntimeError if it can't converge
        raise HTTPException(status_code=422, detail=f"Forecast model failed to fit: {e}")

    return PredictResponse(**result)