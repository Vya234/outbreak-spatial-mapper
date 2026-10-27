"""
Phase 3 - Real-World Prediction API
-----------------------------------
A FastAPI service connecting the Leaflet frontend with SIR optimization
and outbreak data ingestion.

Note:
- The SIR forecast is a prototype, not a clinically validated prediction.
- The spatial radius calculation is heuristic and requires independent
  validation against held-out geographic outbreak observations.
"""

import os
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_engineering import compute_daily_features
from forecasting_engine import (
    optimize_sir_parameters,
    forecast_sir_cumulative,
)
from dataset_loader import OutbreakDatasetLoader


# ---------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------

app = FastAPI(
    title="Outbreak Prediction API",
    version="3.1.0",
    description="Prototype spatial and epidemiological outbreak forecasting API.",
)


# ---------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://outbreak-spatial-mapper.vercel.app",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------

class CaseIn(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    day: int = Field(..., ge=1)


class PredictRequest(BaseModel):
    cases: List[CaseIn] = Field(..., min_length=2)
    target_day: int = Field(..., ge=1)
    population_estimate: int = Field(
        default=50000,
        ge=1000,
        description="Initial population assumption for SIR parameter optimization.",
    )


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


# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """
    Check for the optional preprocessed dataset.

    The prediction endpoint can still operate on coordinates supplied
    directly in a /predict request when this dataset is unavailable.
    """
    os.makedirs("data", exist_ok=True)

    processed_path = os.path.join(
        "data",
        "processed_dengue_mh.csv",
    )

    if not os.path.exists(processed_path):
        print(
            "Optional processed dengue dataset not found. "
            "Dataset-backed endpoints may be unavailable; "
            "coordinate-based prediction remains enabled."
        )


# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "system": "OutbreakIQ Phase 3 Active",
    }


# ---------------------------------------------------------------------
# Real outbreak data
# ---------------------------------------------------------------------

@app.get("/api/real-outbreaks")
def get_real_outbreaks():
    """
    Return up to 500 records from the optional preprocessed dataset.
    """
    loader = OutbreakDatasetLoader(data_dir="data")

    try:
        df = loader.load_dataset("processed_dengue_mh.csv")
        return df.head(500).to_dict(orient="records")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load the outbreak dataset: {exc}",
        ) from exc


# ---------------------------------------------------------------------
# Geographic helper
# ---------------------------------------------------------------------

def calculate_spread_radius_km(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    centroid_lat: float,
    centroid_lng: float,
) -> float:
    """
    Calculate the maximum approximate great-circle distance from the
    centroid to an observed point.

    This uses the Haversine formula rather than treating one degree of
    latitude and longitude as having the same physical distance.
    """
    earth_radius_km = 6371.0088

    latitudes_rad = np.radians(latitudes)
    longitudes_rad = np.radians(longitudes)

    centroid_lat_rad = np.radians(centroid_lat)
    centroid_lng_rad = np.radians(centroid_lng)

    delta_lat = latitudes_rad - centroid_lat_rad
    delta_lng = longitudes_rad - centroid_lng_rad

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(centroid_lat_rad)
        * np.cos(latitudes_rad)
        * np.sin(delta_lng / 2.0) ** 2
    )

    # Protect against floating-point values slightly outside [0, 1].
    a = np.clip(a, 0.0, 1.0)

    distances_km = (
        2.0
        * earth_radius_km
        * np.arcsin(np.sqrt(a))
    )

    return float(np.max(distances_km))


# ---------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict a future cumulative SIR case count and estimate a heuristic
    spatial expansion radius.

    Input:
      - cases: observed coordinates and their timeline days
      - target_day: absolute timeline day to forecast
      - population_estimate: initial population assumption for fitting

    Important:
      - Timeline days are assumed to share one consistent origin.
      - Each input coordinate is treated as one logged event.
      - Logged coordinates are not necessarily equivalent to new
        epidemiological infections.
    """

    # -------------------------------------------------------------
    # 1. Validate the requested forecast horizon
    # -------------------------------------------------------------

    max_observed_day = max(case.day for case in req.cases)

    if req.target_day <= max_observed_day:
        raise HTTPException(
            status_code=400,
            detail=(
                f"target_day ({req.target_day}) must be after "
                f"the latest observed day ({max_observed_day})."
            ),
        )

    # -------------------------------------------------------------
    # 2. Prepare coordinate observations
    # -------------------------------------------------------------

    df = pd.DataFrame(
        [
            {
                "Point_ID": index,
                "Latitude": case.lat,
                "Longitude": case.lng,
                "Timeline_Day": case.day,
            }
            for index, case in enumerate(req.cases, start=1)
        ]
    )

    # Build daily features. Missing timeline days should represent
    # days with no newly logged coordinates, not confirmed zero cases.
    try:
        daily = compute_daily_features(
            df,
            cumulative=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not construct daily features: {exc}",
        ) from exc

    if daily.empty or daily["day"].nunique() < 3:
        raise HTTPException(
            status_code=400,
            detail=(
                "Need observations spanning at least 3 distinct "
                "timeline days to fit the forecast."
            ),
        )

    # -------------------------------------------------------------
    # 3. Fit SIR parameters and forecast the cumulative trajectory
    # -------------------------------------------------------------

    try:
        if "newly_logged" not in daily.columns:
            raise ValueError(
                "Daily features must contain a 'newly_logged' column."
            )

        daily_counts = daily["newly_logged"].to_numpy(dtype=float)

        if (
            not np.all(np.isfinite(daily_counts))
            or np.any(daily_counts < 0)
        ):
            raise ValueError(
                "Daily logged counts must be finite and non-negative."
            )

        # Fit the SIR parameters to the daily input series.
        opt_res = optimize_sir_parameters(
            daily_counts,
            initial_population=req.population_estimate,
        )

        if not opt_res.get("success", False):
            raise ValueError(
                "SIR parameter optimizer did not converge successfully."
            )

        opt_N = float(opt_res["optimized_N"])
        opt_beta = float(opt_res["optimized_beta"])
        opt_gamma = float(opt_res["optimized_gamma"])

        if opt_N <= 0 or opt_beta < 0 or opt_gamma <= 0:
            raise ValueError(
                "The optimizer returned invalid SIR parameters."
            )

        # Continue the SIR trajectory across the observed period and
        # then forecast to target_day. The helper must interpret
        # forecast_horizon as the number of days beyond the end of the
        # daily input series.
        forecast_horizon = req.target_day - max_observed_day

        forecast_result = forecast_sir_cumulative(
            daily_cases=daily_counts,
            forecast_horizon=forecast_horizon,
            population=opt_N,
            beta=opt_beta,
            gamma=opt_gamma,
        )

        # Expected helper result: a numeric cumulative case prediction.
        # If your forecasting_engine.py returns a dictionary instead,
        # adjust this extraction to match its actual return type.
        if isinstance(forecast_result, dict):
            predicted_total_cases = float(
                forecast_result["predicted_cumulative_cases"]
            )
        else:
            predicted_total_cases = float(forecast_result)

        if (
            not np.isfinite(predicted_total_cases)
            or predicted_total_cases < 0
        ):
            raise ValueError(
                "The SIR forecast returned an invalid case count."
            )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Forecast model optimization failed: {exc}",
        ) from exc

    # -------------------------------------------------------------
    # 4. Calculate centroid and observed spatial spread
    # -------------------------------------------------------------

    latitudes = df["Latitude"].to_numpy(dtype=float)
    longitudes = df["Longitude"].to_numpy(dtype=float)

    centroid_lat = float(np.mean(latitudes))
    centroid_lng = float(np.mean(longitudes))

    base_radius_km = calculate_spread_radius_km(
        latitudes=latitudes,
        longitudes=longitudes,
        centroid_lat=centroid_lat,
        centroid_lng=centroid_lng,
    )

    # -------------------------------------------------------------
    # 5. Estimate future spatial radius
    # -------------------------------------------------------------
    # These are heuristic estimates, not a fitted or validated spatial
    # propagation model. The 5% daily growth and 15% uncertainty are
    # assumptions that require calibration against held-out data.

    day_delta = req.target_day - max_observed_day

    radius_trend = max(
        1.0,
        base_radius_km * (1.0 + 0.05 * day_delta),
    )

    sir_multiplier = opt_beta / max(0.001, opt_gamma)

    radius_sir = radius_trend * min(
        3.0,
        max(0.5, sir_multiplier * 0.5),
    )

    radius_uncertainty = radius_sir * 0.15

    # -------------------------------------------------------------
    # 6. Return the prediction
    # -------------------------------------------------------------

    return PredictResponse(
        target_day=req.target_day,
        centroid_lat=round(centroid_lat, 5),
        centroid_lng=round(centroid_lng, 5),
        radius_km_trend_only=round(radius_trend, 2),
        radius_km_sir_scaled=round(radius_sir, 2),
        radius_uncertainty_km=round(radius_uncertainty, 2),
        predicted_case_count=round(predicted_total_cases, 1),
        sir_beta=round(opt_beta, 4),
        sir_gamma=round(opt_gamma, 4),
        sir_population_assumed=int(round(opt_N)),
    )
