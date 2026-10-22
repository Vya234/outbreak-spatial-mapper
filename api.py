"""
Phase 3 - Real-World Prediction API
---------------------------------------
A FastAPI service that bridges the Leaflet frontend with the updated
gradient-based SIR optimization and real-world dataset ingestion.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_engineering import compute_daily_features
from forecasting_engine import optimize_sir_parameters, sir_model_ode
from dataset_loader import OutbreakDatasetLoader
from epiclim_adapter import EpiClimAdapter
from scipy.integrate import odeint

app = FastAPI(title="Outbreak Prediction API", version="3.0.0")

# --- CORS ---------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
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
    population_estimate: int = Field(50000, ge=1, description="Initial population assumption for gradient optimization")


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


# --- Startup Event (Ensures real dataset is prepped on host boot) ---------

@app.on_event("startup")
def startup_event():
    import os

    os.makedirs("data", exist_ok=True)
    processed_path = "data/processed_dengue_mh.csv"

    if not os.path.exists(processed_path):
        print(
            "Processed dengue dataset is missing. "
            "Skipping optional dataset preparation; "
            "the prediction endpoint remains available."
        )


# --- Routes ---------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "system": "OutbreakIQ Phase 3 Active"}


@app.get("/api/real-outbreaks")
def get_real_outbreaks():
    """
    Optional endpoint to serve the preprocessed EpiClim dataset directly to the frontend map.
    """
    loader = OutbreakDatasetLoader(data_dir="data")
    try:
        df = loader.load_dataset("processed_dengue_mh.csv")
        # Return a sample or full dataset records
        return df.head(500).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

    if daily["day"].nunique() < 3:
        raise HTTPException(
            status_code=400,
            detail="Need cases spread across at least 3 distinct days to fit a trend.",
        )

    try:
        # Extract daily case counts for gradient optimization
        daily_counts = daily["new_cases"].values if "new_cases" in daily.columns else daily.iloc[:, 1].diff().fillna(1).values
        
        # Run the new gradient-based optimization engine
        opt_res = optimize_sir_parameters(daily_counts, initial_population=req.population_estimate)
        
        if not opt_res["success"]:
            raise RuntimeError("Gradient optimizer failed to converge.")

        opt_N = opt_res["optimized_N"]
        opt_beta = opt_res["optimized_beta"]
        opt_gamma = opt_res["optimized_gamma"]

        # Compute spatial centroid and spread radius
        centroid_lat = float(df["Latitude"].mean())
        centroid_lng = float(df["Longitude"].mean())
        
        # Trend-only & SIR-scaled spatial radius estimation
        max_dist_deg = np.max(np.sqrt((df["Latitude"] - centroid_lat)**2 + (df["Longitude"] - centroid_lng)**2))
        base_radius_km = max_dist_deg * 111.0 # rough lat/lng to km conversion
        day_delta = req.target_day - max_observed_day
        
        radius_trend = max(1.0, base_radius_km * (1.0 + 0.05 * day_delta))
        sir_multiplier = opt_beta / max(0.001, opt_gamma)
        radius_sir = radius_trend * min(3.0, max(0.5, sir_multiplier * 0.5))

        # Predict future cumulative cases using optimized ODE parameters
        t_forecast = np.arange(0, req.target_day + 1)
        cumulative_actual = np.cumsum(daily_counts)
        I0 = max(1.0, cumulative_actual[0])
        S0 = opt_N - I0
        solution = odeint(sir_model_ode, [S0, I0, 0.0], t_forecast, args=(opt_N, opt_beta, opt_gamma))
        predicted_total_cases = float(solution[-1, 1] + solution[-1, 2])

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Forecast model optimization failed: {e}")

    return PredictResponse(
        target_day=req.target_day,
        centroid_lat=round(centroid_lat, 5),
        centroid_lng=round(centroid_lng, 5),
        radius_km_trend_only=round(radius_trend, 2),
        radius_km_sir_scaled=round(radius_sir, 2),
        radius_uncertainty_km=round(radius_sir * 0.15, 2),
        predicted_case_count=round(predicted_total_cases, 1),
        sir_beta=opt_beta,
        sir_gamma=opt_gamma,
        sir_population_assumed=int(opt_N)
    )
