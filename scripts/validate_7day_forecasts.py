import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_engineering import compute_daily_features


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = ROOT / "data" / "processed_dengue_mh.csv"
OUTPUT_DIR = ROOT / "research_outputs"

MIN_TRAIN_SIZE = 120
FORECAST_HORIZON = 7


# --------------------------------------------------
# Load and prepare data
# --------------------------------------------------

def load_daily_counts():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    daily_features = compute_daily_features(df, cumulative=True)

    if "newly_logged" not in daily_features.columns:
        raise ValueError(
            "Expected 'newly_logged' column from compute_daily_features()."
        )

    daily_counts = pd.to_numeric(
        daily_features["newly_logged"], errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(daily_counts).all():
        raise ValueError("Daily counts contain missing or non-finite values.")

    if (daily_counts < 0).any():
        raise ValueError("Daily counts cannot be negative.")

    return daily_features.reset_index(drop=True), daily_counts


# --------------------------------------------------
# Forecasting methods
# --------------------------------------------------

def generate_forecasts(history):
    """
    Generate forecasts using only observations available
    before the current forecast origin.

    Every method returns one constant value for the
    complete 7-day forecast horizon.
    """

    if len(history) == 0:
        raise ValueError("History cannot be empty.")

    forecasts = {
        "Expanding Training Median": float(np.median(history)),
        "Persistence (Last Value)": float(history[-1]),
        "Recent 7-Day Median": float(np.median(history[-7:])),
        "Recent 7-Day Mean": float(np.mean(history[-7:])),
    }

    return forecasts


# --------------------------------------------------
# Rolling-origin validation
# --------------------------------------------------

def run_validation(daily_features, daily_counts):
    total_days = len(daily_counts)

    if total_days <= MIN_TRAIN_SIZE:
        raise ValueError(
            f"Need more than {MIN_TRAIN_SIZE} daily observations; "
            f"found {total_days}."
        )

    detailed_results = []
    origin_summaries = []

    origin = MIN_TRAIN_SIZE
    block_number = 1

    while origin < total_days:
        train = daily_counts[:origin]

        test_end = min(origin + FORECAST_HORIZON, total_days)
        actual_values = daily_counts[origin:test_end]

        if len(actual_values) == 0:
            break

        forecasts = generate_forecasts(train)

        # Evaluate every model over this forecast block.
        for model_name, prediction in forecasts.items():
            predictions = np.full(len(actual_values), prediction)

            block_mae = mean_absolute_error(
                actual_values, predictions
            )
            block_rmse = np.sqrt(
                mean_squared_error(actual_values, predictions)
            )

            origin_summaries.append({
                "block": block_number,
                "origin_index": origin,
                "training_days": len(train),
                "forecast_start_index": origin,
                "forecast_end_index": test_end - 1,
                "forecast_days": len(actual_values),
                "model": model_name,
                "MAE": block_mae,
                "RMSE": block_rmse,
            })

            for offset, (actual, predicted) in enumerate(
                zip(actual_values, predictions)
            ):
                day_index = origin + offset

                row = {
                    "block": block_number,
                    "origin_index": origin,
                    "day_index": day_index,
                    "actual_logged_records": float(actual),
                    "model": model_name,
                    "predicted_logged_records": float(predicted),
                    "absolute_error": float(abs(actual - predicted)),
                    "squared_error": float((actual - predicted) ** 2),
                }

                # Include the corresponding timeline day if available.
                if "day" in daily_features.columns:
                    row["timeline_day"] = daily_features.iloc[
                        day_index
                    ]["day"]

                detailed_results.append(row)

        origin += FORECAST_HORIZON
        block_number += 1

    return pd.DataFrame(detailed_results), pd.DataFrame(origin_summaries)


# --------------------------------------------------
# Aggregate metrics
# --------------------------------------------------

def calculate_summary(detailed_df, block_df):
    """
    Report both pooled metrics across all forecasted days
    and the average error across forecast blocks.
    """

    summaries = []

    for model_name, group in detailed_df.groupby("model", sort=False):
        actual = group["actual_logged_records"].to_numpy()
        predicted = group["predicted_logged_records"].to_numpy()

        model_blocks = block_df[block_df["model"] == model_name]

        summaries.append({
            "model": model_name,
            "forecast_blocks": int(model_blocks["block"].nunique()),
            "total_test_days": len(group),
            "pooled_MAE": mean_absolute_error(actual, predicted),
            "pooled_RMSE": np.sqrt(
                mean_squared_error(actual, predicted)
            ),
            "mean_block_MAE": model_blocks["MAE"].mean(),
            "mean_block_RMSE": model_blocks["RMSE"].mean(),
            "median_block_MAE": model_blocks["MAE"].median(),
            "median_block_RMSE": model_blocks["RMSE"].median(),
        })

    summary_df = pd.DataFrame(summaries)

    # Rank methods by pooled MAE. Lower is better.
    summary_df = summary_df.sort_values(
        by=["pooled_MAE", "pooled_RMSE"]
    ).reset_index(drop=True)

    summary_df.insert(
        0, "rank_by_pooled_MAE", np.arange(1, len(summary_df) + 1)
    )

    return summary_df


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_features, daily_counts = load_daily_counts()

    print("=" * 65)
    print("7-DAY FORECAST VALIDATION")
    print("=" * 65)
    print(f"Dataset: {DATA_PATH}")
    print(f"Total daily observations: {len(daily_counts)}")
    print(f"Minimum training observations: {MIN_TRAIN_SIZE}")
    print(f"Forecast horizon: {FORECAST_HORIZON} days")
    print("Target: daily logged-record counts")
    print()

    detailed_df, block_df = run_validation(
        daily_features, daily_counts
    )

    summary_df = calculate_summary(detailed_df, block_df)

    detailed_path = (
        OUTPUT_DIR / "7day_forecast_detailed_results.csv"
    )
    blocks_path = (
        OUTPUT_DIR / "7day_forecast_block_metrics.csv"
    )
    summary_path = (
        OUTPUT_DIR / "7day_forecast_summary.csv"
    )

    detailed_df.to_csv(detailed_path, index=False)
    block_df.to_csv(blocks_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("SUMMARY — LOWER ERRORS ARE BETTER")
    print("-" * 65)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}"
        )
    )

    print()
    print(f"Detailed predictions saved to: {detailed_path}")
    print(f"Block-level metrics saved to: {blocks_path}")
    print(f"Summary saved to: {summary_path}")

    best_model = summary_df.iloc[0]

    print()
    print("=" * 65)
    print("BEST MODEL BY POOLED MAE")
    print("=" * 65)
    print(f"Model: {best_model['model']}")
    print(f"Pooled MAE: {best_model['pooled_MAE']:.3f}")
    print(f"Pooled RMSE: {best_model['pooled_RMSE']:.3f}")
    print()
    print(
        "Interpretation: these metrics evaluate predictions of daily "
        "logged-record counts, not necessarily actual new infections."
    )


if __name__ == "__main__":
    main()
