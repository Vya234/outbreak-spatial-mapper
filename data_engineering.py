"""
Phase 2 - Step 1: Data Engineering
-----------------------------------
Turns the raw case export (Point_ID, Latitude, Longitude, Timeline_Day)
from the Outbreak Tracker into a daily time-series of geographic
features (centroid + spread radius) that the forecasting model in
Step 2 will consume.

Key design choice: "active cases on day N" is treated as CUMULATIVE
(every case logged on or before day N), matching how the Leaflet
frontend's time slider already works (case.day <= currentDayLimit).
Pass cumulative=False if you'd rather look at only the cases newly
reported on each day.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------

def load_cases(csv_path: str) -> pd.DataFrame:
    """Load the exported case CSV and do basic validation/cleanup."""
    df = pd.read_csv(csv_path)

    expected_cols = {"Point_ID", "Latitude", "Longitude", "Timeline_Day"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    df = df.dropna(subset=["Latitude", "Longitude", "Timeline_Day"]).copy()
    df["Timeline_Day"] = df["Timeline_Day"].astype(int)
    df = df.sort_values("Timeline_Day").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# 2. Geometry helpers
# ---------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lng points, in km."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def centroid(points: pd.DataFrame) -> tuple[float, float]:
    """Simple mean-lat/mean-lng centroid (fine at neighbourhood scale)."""
    return points["Latitude"].mean(), points["Longitude"].mean()


def spread_radius_km(points: pd.DataFrame, centroid_lat: float, centroid_lng: float) -> float:
    """Max distance from centroid to any case = radius of the outbreak."""
    if len(points) == 0:
        return 0.0
    dists = points.apply(
        lambda r: haversine_km(centroid_lat, centroid_lng, r["Latitude"], r["Longitude"]),
        axis=1,
    )
    return float(dists.max())


# ---------------------------------------------------------------------
# 3. Daily feature aggregation
# ---------------------------------------------------------------------

@dataclass
class DailyFeatures:
    day: int
    case_count: int
    centroid_lat: float
    centroid_lng: float
    spread_radius_km: float


def compute_daily_features(df: pd.DataFrame, cumulative: bool = True) -> pd.DataFrame:
    """
    Build one row per day in [min_day, max_day] with:
      day, case_count, centroid_lat, centroid_lng, spread_radius_km

    cumulative=True  -> "active cases up to and including this day"
                         (matches the app's time slider behaviour)
    cumulative=False -> only cases newly logged on that exact day
    """
    min_day, max_day = int(df["Timeline_Day"].min()), int(df["Timeline_Day"].max())
    rows: list[DailyFeatures] = []

    for day in range(min_day, max_day + 1):
        subset = df[df["Timeline_Day"] <= day] if cumulative else df[df["Timeline_Day"] == day]

        if len(subset) == 0:
            rows.append(DailyFeatures(day, 0, np.nan, np.nan, 0.0))
            continue

        c_lat, c_lng = centroid(subset)
        radius = spread_radius_km(subset, c_lat, c_lng)
        rows.append(DailyFeatures(day, len(subset), c_lat, c_lng, radius))

    out = pd.DataFrame([r.__dict__ for r in rows])

    # forward-fill centroid on any empty days (only relevant for
    # cumulative=False, where a day might have zero new cases)
    out[["centroid_lat", "centroid_lng"]] = out[["centroid_lat", "centroid_lng"]].ffill()
    return out


# ---------------------------------------------------------------------
# 4. Sequence generation (input/output pairs for Step 2's model)
# ---------------------------------------------------------------------

def generate_sequences(
    daily_features: pd.DataFrame,
    seq_len: int = 7,
    horizon: int = 7,
    feature_cols: tuple[str, ...] = ("centroid_lat", "centroid_lng", "spread_radius_km", "case_count"),
):
    """
    Slide a window of `seq_len` days across the time-series to build
    (X, y) pairs: X = seq_len days of features, y = the same features
    `horizon` days later.

    With only ~14 days of data from a single synthetic outbreak this
    will yield very few (or zero) windows -- that's expected, and is
    exactly why Step 2 uses a simple regression fit on the trend
    rather than a data-hungry model like an LSTM.
    """
    values = daily_features[list(feature_cols)].to_numpy()
    n = len(values)

    X, y = [], []
    for start in range(0, n - seq_len - horizon + 1):
        X.append(values[start : start + seq_len])
        y.append(values[start + seq_len + horizon - 1])

    X = np.array(X)
    y = np.array(y)
    return X, y


# ---------------------------------------------------------------------
# 5. Demo / smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic export matching the app's default seed cases, so the
    # pipeline can be sanity-checked before you drop in a real export.
    demo_rows = [
        (1, 22.3149, 87.3105, 1),
        (2, 22.3155, 87.3151, 3),
        (3, 22.3121, 87.3120, 5),
        (4, 22.3160, 87.3110, 8),
        (5, 22.3135, 87.3135, 10),
        (6, 22.3142, 87.3098, 11),
        (7, 22.3168, 87.3140, 12),
        (8, 22.3110, 87.3145, 13),
        (9, 22.3175, 87.3122, 14),
    ]
    demo_df = pd.DataFrame(demo_rows, columns=["Point_ID", "Latitude", "Longitude", "Timeline_Day"])
    demo_df.to_csv("/tmp/demo_export.csv", index=False)

    cases = load_cases("/tmp/demo_export.csv")
    daily = compute_daily_features(cases, cumulative=True)
    print("Daily features (cumulative):")
    print(daily.to_string(index=False))

    X, y = generate_sequences(daily, seq_len=7, horizon=1)
    print(f"\nSequence windows generated: {X.shape[0]} (shape X={X.shape}, y={y.shape})")