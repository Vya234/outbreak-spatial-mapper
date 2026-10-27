import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_engineering import compute_daily_features


DATA_PATH = PROJECT_ROOT / "data" / "processed_dengue_mh.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "research_outputs"
    / "count_baseline_rolling_validation.csv"
)

# Evaluate each forecast origin over the following 30 observations.
TEST_HORIZON = 30

# Require enough training history before evaluating a forecast.
MIN_TRAIN_SIZE = 120

WINDOWS = [3, 7, 14, 28]


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return float(mae), float(rmse)


def main():
    df = pd.read_csv(DATA_PATH)

    daily = compute_daily_features(df, cumulative=True)
    daily = daily.sort_values("day").reset_index(drop=True)

    counts = daily["newly_logged"].to_numpy(dtype=float)

    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Daily counts contain invalid values.")

    if len(counts) < MIN_TRAIN_SIZE + TEST_HORIZON:
        raise ValueError("Not enough observations for rolling validation.")

    # Use expanding training windows and non-overlapping test blocks.
    # Every prediction is based only on observations before its origin.
    origins = list(
        range(
            MIN_TRAIN_SIZE,
            len(counts) - TEST_HORIZON + 1,
            TEST_HORIZON,
        )
    )

    records = []

    for origin in origins:
        train = counts[:origin]
        test = counts[origin:origin + TEST_HORIZON]

        predictions = {
            "Training median": np.full(
                len(test), np.median(train)
            ),
            "Training mean": np.full(
                len(test), np.mean(train)
            ),
        }

        for window in WINDOWS:
            predictions[f"Recent {window}-day median"] = np.full(
                len(test),
                np.median(train[-window:]),
            )

        for model_name, predicted in predictions.items():
            mae, rmse = calculate_metrics(test, predicted)

            records.append({
                "origin_index": origin,
                "train_size": len(train),
                "test_start_day": daily["day"].iloc[origin],
                "test_end_day": daily["day"].iloc[origin + len(test) - 1],
                "model": model_name,
                "mae": mae,
                "rmse": rmse,
                "test_observations": len(test),
            })

    detailed = pd.DataFrame(records)

    summary = (
        detailed.groupby("model")
        .agg(
            forecast_blocks=("origin_index", "nunique"),
            total_test_observations=("test_observations", "sum"),
            mean_block_mae=("mae", "mean"),
            mean_block_rmse=("rmse", "mean"),
            median_block_mae=("mae", "median"),
            median_block_rmse=("rmse", "median"),
        )
        .reset_index()
        .sort_values("mean_block_mae")
    )

    print("=" * 75)
    print("ROLLING-ORIGIN COUNT BASELINE VALIDATION")
    print("=" * 75)
    print("Total observations:", len(counts))
    print("Minimum training size:", MIN_TRAIN_SIZE)
    print("Test horizon per block:", TEST_HORIZON)
    print("Forecast origins:", len(origins))

    print("\nSUMMARY (lower errors are better)")
    print(
        summary.to_string(
            index=False,
            formatters={
                column: "{:.3f}".format
                for column in [
                    "mean_block_mae",
                    "mean_block_rmse",
                    "median_block_mae",
                    "median_block_rmse",
                ]
            },
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    detailed_path = OUTPUT_PATH
    summary_path = OUTPUT_PATH.with_name(
        "count_baseline_rolling_summary.csv"
    )

    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nSaved detailed results to:", detailed_path)
    print("Saved summary to:", summary_path)


if __name__ == "__main__":
    main()
