import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed_dengue_mh.csv")
OUTPUT_DIR = Path("research_outputs")

FORECAST_HORIZON = 7
MIN_HISTORY_DAYS = 30
STEP_SIZE = 7
EARTH_RADIUS_KM = 6371.0088


def haversine_distances_km(latitudes, longitudes, center_lat, center_lng):
    """Calculate great-circle distances from points to a fixed center."""
    latitudes = np.radians(np.asarray(latitudes, dtype=float))
    longitudes = np.radians(np.asarray(longitudes, dtype=float))
    center_lat = np.radians(center_lat)
    center_lng = np.radians(center_lng)

    delta_lat = latitudes - center_lat
    delta_lng = longitudes - center_lng

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(center_lat)
        * np.cos(latitudes)
        * np.sin(delta_lng / 2) ** 2
    )
    a = np.clip(a, 0.0, 1.0)

    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def main():
    df = pd.read_csv(DATA_PATH)

    required = {"Latitude", "Longitude", "Timeline_Day"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(
        subset=["Latitude", "Longitude", "Timeline_Day"]
    ).copy()

    df["Timeline_Day"] = df["Timeline_Day"].astype(int)
    df = df.sort_values("Timeline_Day")

    days = np.sort(df["Timeline_Day"].unique())

    detailed_results = []
    block_results = []

    # Each origin uses only records observed on or before that day.
    for origin_index in range(
        MIN_HISTORY_DAYS - 1,
        len(days) - 1,
        STEP_SIZE,
    ):
        origin_day = days[origin_index]
        future_days = days[
            origin_index + 1:
            origin_index + 1 + FORECAST_HORIZON
        ]

        if len(future_days) == 0:
            break

        history = df[df["Timeline_Day"] <= origin_day]
        future = df[df["Timeline_Day"].isin(future_days)]

        if history.empty or future.empty:
            continue

        # Historical centroid and radius use historical data only.
        center_lat = float(history["Latitude"].mean())
        center_lng = float(history["Longitude"].mean())

        historical_distances = haversine_distances_km(
            history["Latitude"],
            history["Longitude"],
            center_lat,
            center_lng,
        )
        base_radius_km = float(np.max(historical_distances))

        future_distances = haversine_distances_km(
            future["Latitude"],
            future["Longitude"],
            center_lat,
            center_lng,
        )

        actual_future_points = len(future)

        methods = {
            "No Expansion": base_radius_km,
            "5% Daily Expansion": base_radius_km
            * (1 + 0.05 * len(future_days)),
            "10% Daily Expansion": base_radius_km
            * (1 + 0.10 * len(future_days)),
        }

        for method, predicted_radius_km in methods.items():
            covered = future_distances <= predicted_radius_km
            coverage = float(np.mean(covered))
            area_km2 = np.pi * predicted_radius_km**2

            detailed_results.append({
                "origin_day": int(origin_day),
                "forecast_start_day": int(future_days[0]),
                "forecast_end_day": int(future_days[-1]),
                "horizon_days": len(future_days),
                "method": method,
                "historical_points": len(history),
                "future_points": actual_future_points,
                "historical_radius_km": base_radius_km,
                "predicted_radius_km": predicted_radius_km,
                "predicted_circle_area_km2": area_km2,
                "covered_future_points": int(np.sum(covered)),
                "coverage_fraction": coverage,
            })

            block_results.append({
                "origin_day": int(origin_day),
                "method": method,
                "future_points": actual_future_points,
                "covered_future_points": int(np.sum(covered)),
                "coverage_fraction": coverage,
                "predicted_radius_km": predicted_radius_km,
                "predicted_circle_area_km2": area_km2,
            })

    detailed = pd.DataFrame(detailed_results)
    if detailed.empty:
        raise RuntimeError(
            "No validation windows were created. Check the timeline."
        )

    detailed.to_csv(
        OUTPUT_DIR / "spatial_coverage_detailed.csv",
        index=False,
    )

    # Pooled coverage weights each future coordinate record equally.
    summary = (
        detailed.groupby("method")
        .apply(
            lambda group: pd.Series({
                "forecast_windows": len(group),
                "future_points": int(group["future_points"].sum()),
                "pooled_coverage": (
                    group["covered_future_points"].sum()
                    / group["future_points"].sum()
                ),
                "mean_window_coverage": group[
                    "coverage_fraction"
                ].mean(),
                "median_window_coverage": group[
                    "coverage_fraction"
                ].median(),
                "mean_predicted_radius_km": group[
                    "predicted_radius_km"
                ].mean(),
                "mean_predicted_circle_area_km2": group[
                    "predicted_circle_area_km2"
                ].mean(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "spatial_coverage_summary.csv",
        index=False,
    )

    print("\nSPATIAL COVERAGE VALIDATION")
    print("===========================")
    print(f"Dataset records: {len(df)}")
    print(f"Timeline days: {len(days)}")
    print(f"Forecast horizon: up to {FORECAST_HORIZON} days")
    print(f"Validation windows: {detailed['origin_day'].nunique()}")

    print("\nSummary:")
    print(summary.round(4).to_string(index=False))

    print("\nFirst 6 validation windows:")
    print(detailed.head(6).round(4).to_string(index=False))

    print("\nFiles saved:")
    print(" - research_outputs/spatial_coverage_detailed.csv")
    print(" - research_outputs/spatial_coverage_summary.csv")


if __name__ == "__main__":
    main()
