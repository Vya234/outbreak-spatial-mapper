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

Pass cumulative=False to calculate features for locations logged
on each individual day.

Output columns:
- day
- case_count
- centroid_lat
- centroid_lng
- spread_radius_km
- newly_logged

Important:
"newly_logged" counts newly logged locations, not necessarily new
epidemiological infections.
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

    expected_cols = {
        "Point_ID",
        "Latitude",
        "Longitude",
        "Timeline_Day",
    }

    missing = expected_cols - set(df.columns)

    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    df = df.dropna(
        subset=["Latitude", "Longitude", "Timeline_Day"]
    ).copy()

    df["Timeline_Day"] = df["Timeline_Day"].astype(int)
    df = df.sort_values("Timeline_Day").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# 2. Geometry helpers
# ---------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lng points, in km."""
    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2],
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    # Protect against small floating-point rounding errors.
    a = min(1.0, max(0.0, a))

    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def centroid(points: pd.DataFrame) -> tuple[float, float]:
    """Simple mean-latitude/mean-longitude centroid."""
    return (
        float(points["Latitude"].mean()),
        float(points["Longitude"].mean()),
    )


def spread_radius_km(
    points: pd.DataFrame,
    centroid_lat: float,
    centroid_lng: float,
) -> float:
    """Maximum great-circle distance from the centroid, in km."""
    if len(points) == 0:
        return 0.0

    distances = points.apply(
        lambda row: haversine_km(
            centroid_lat,
            centroid_lng,
            row["Latitude"],
            row["Longitude"],
        ),
        axis=1,
    )

    return float(distances.max())


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


def compute_daily_features(
    df: pd.DataFrame,
    cumulative: bool = True,
) -> pd.DataFrame:
    """
    Build one row per day in [min_day, max_day].

    cumulative=True:
        case_count is the total number of locations logged on or
        before that day.

    cumulative=False:
        case_count is the number of locations logged on that day.

    newly_logged:
        Number of locations first logged on each day. This is
        calculated explicitly in both modes.

    Days with no new locations have newly_logged=0. Centroids are
    forward-filled on empty days where a previous centroid exists.
    """
    output_columns = [
        "day",
        "case_count",
        "centroid_lat",
        "centroid_lng",
        "spread_radius_km",
        "newly_logged",
    ]

    if df.empty:
        return pd.DataFrame(columns=output_columns)

    required_cols = {
        "Latitude",
        "Longitude",
        "Timeline_Day",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}"
        )

    working_df = df.dropna(
        subset=["Latitude", "Longitude", "Timeline_Day"]
    ).copy()

    if working_df.empty:
        return pd.DataFrame(columns=output_columns)

    working_df["Timeline_Day"] = (
        working_df["Timeline_Day"].astype(int)
    )

    min_day = int(working_df["Timeline_Day"].min())
    max_day = int(working_df["Timeline_Day"].max())

    rows: list[DailyFeatures] = []

    # Process EVERY day in the timeline.
    for day in range(min_day, max_day + 1):
        if cumulative:
            subset = working_df[
                working_df["Timeline_Day"] <= day
            ]
        else:
            subset = working_df[
                working_df["Timeline_Day"] == day
            ]

        if subset.empty:
            rows.append(
                DailyFeatures(
                    day=day,
                    case_count=0,
                    centroid_lat=np.nan,
                    centroid_lng=np.nan,
                    spread_radius_km=0.0,
                )
            )
            continue

        center_lat, center_lng = centroid(subset)

        radius = spread_radius_km(
            subset,
            center_lat,
            center_lng,
        )

        rows.append(
            DailyFeatures(
                day=day,
                case_count=len(subset),
                centroid_lat=center_lat,
                centroid_lng=center_lng,
                spread_radius_km=radius,
            )
        )

    # IMPORTANT: DataFrame creation and return are OUTSIDE the loop.
    out = pd.DataFrame(
        [row.__dict__ for row in rows]
    )

    # Explicitly calculate newly logged locations per day.
    if cumulative:
        out["newly_logged"] = (
            out["case_count"]
            .diff()
            .fillna(out["case_count"].iloc[0])
            .astype(int)
        )
    else:
        out["newly_logged"] = (
            out["case_count"].astype(int)
        )

    # Keep centroids continuous through days without new locations.
    out[["centroid_lat", "centroid_lng"]] = (
        out[["centroid_lat", "centroid_lng"]].ffill()
    )

    return out[output_columns]


# ---------------------------------------------------------------------
# 4. Sequence generation (input/output pairs for Step 2's model)
# ---------------------------------------------------------------------

def generate_sequences(
    daily_features: pd.DataFrame,
    seq_len: int = 7,
    horizon: int = 7,
    feature_cols: tuple[str, ...] = (
        "centroid_lat",
        "centroid_lng",
        "spread_radius_km",
        "case_count",
    ),
):
    """
    Slide a window of `seq_len` days across the time series.

    X contains seq_len consecutive days of features.
    y contains the feature values horizon days after the input window.

    With only ~14 days of data from a single synthetic outbreak,
    this produces very few windows. That is why Step 2 uses a simple
    trend fit rather than a data-hungry model like an LSTM.
    """
    values = daily_features[list(feature_cols)].to_numpy()
    n = len(values)

    X, y = [], []

    for start in range(0, n - seq_len - horizon + 1):
        X.append(
            values[start:start + seq_len]
        )

        y.append(
            values[start + seq_len + horizon - 1]
        )

    return np.array(X), np.array(y)


# ---------------------------------------------------------------------
# 5. Demo / smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic export matching the app's default seed cases.
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

    demo_df = pd.DataFrame(
        demo_rows,
        columns=[
            "Point_ID",
            "Latitude",
            "Longitude",
            "Timeline_Day",
        ],
    )

    demo_df.to_csv(
        "/tmp/demo_export.csv",
        index=False,
    )

    cases = load_cases("/tmp/demo_export.csv")

    daily = compute_daily_features(
        cases,
        cumulative=True,
    )

    print("Daily features (cumulative):")
    print(daily.to_string(index=False))

    X, y = generate_sequences(
        daily,
        seq_len=7,
        horizon=1,
    )

    print(
        f"\nSequence windows generated: "
        f"{X.shape[0]} (shape X={X.shape}, y={y.shape})"
    )