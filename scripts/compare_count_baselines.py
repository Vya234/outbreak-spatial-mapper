import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_engineering import compute_daily_features


DATA_PATH = PROJECT_ROOT / "data" / "processed_dengue_mh.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "research_outputs"
    / "count_baseline_comparison.csv"
)

TRAIN_FRACTION = 0.70
ROLLING_WINDOWS = [3, 7, 14, 28]


def calculate_metrics(actual, predicted):
    """Calculate MAE and RMSE."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths do not match.")

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return float(mae), float(rmse)


def main():
    # --------------------------------------------------
    # 1. Load and prepare daily logged-record counts
    # --------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    daily = compute_daily_features(df, cumulative=True)
    daily = daily.sort_values("day").reset_index(drop=True)

    counts = daily["newly_logged"].to_numpy(dtype=float)

    if not np.all(np.isfinite(counts)):
        raise ValueError("Daily counts contain NaN or infinite values.")

    if np.any(counts < 0):
        raise ValueError("Daily counts contain negative values.")

    split = int(len(counts) * TRAIN_FRACTION)

    train = counts[:split]
    test = counts[split:]

    if len(train) < max(ROLLING_WINDOWS):
        raise ValueError("Not enough training observations for all windows.")

    if len(test) == 0:
        raise ValueError("The test set is empty.")

    print("=" * 65)
    print("COUNT BASELINE COMPARISON")
    print("=" * 65)
    print("Total observations:", len(counts))
    print("Training observations:", len(train))
    print("Testing observations:", len(test))
    print(
        "Training timeline:",
        daily["day"].iloc[0],
        "to",
        daily["day"].iloc[split - 1],
    )
    print(
        "Testing timeline:",
        daily["day"].iloc[split],
        "to",
        daily["day"].iloc[-1],
    )

    # --------------------------------------------------
    # 2. Build constant baseline predictions
    # --------------------------------------------------
    predictions = {
        "Training median": np.full(len(test), np.median(train)),
        "Training mean": np.full(len(test), np.mean(train)),
    }

    # --------------------------------------------------
    # 3. Build recent-window median baselines
    # --------------------------------------------------
    # Each baseline uses only the final observations from
    # the training period, then predicts that constant
    # value throughout the test period.
    for window in ROLLING_WINDOWS:
        recent_median = np.median(train[-window:])

        predictions[f"Recent {window}-day median"] = np.full(
            len(test),
            recent_median,
        )

    # --------------------------------------------------
    # 4. Evaluate every baseline on the same test period
    # --------------------------------------------------
    rows = []

    for model_name, predicted in predictions.items():
        mae, rmse = calculate_metrics(test, predicted)

        rows.append({
            "model": model_name,
            "mae": mae,
            "rmse": rmse,
            "test_observations": len(test),
            "prediction_value": float(predicted[0]),
        })

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["mae", "rmse"]
    ).reset_index(drop=True)

    print("\nRESULTS (lower MAE and RMSE are better)")
    print(
        results.to_string(
            index=False,
            formatters={
                "mae": "{:.3f}".format,
                "rmse": "{:.3f}".format,
                "prediction_value": "{:.3f}".format,
            },
        )
    )

    # --------------------------------------------------
    # 5. Save the comparison
    # --------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nBest model by MAE:", results.iloc[0]["model"])
    print("Saved results to:", OUTPUT_PATH)

    # --------------------------------------------------
    # 6. Print the recent-window values for transparency
    # --------------------------------------------------
    print("\nTRAINING STATISTICS")
    print("Training median:", round(float(np.median(train)), 3))
    print("Training mean:", round(float(np.mean(train)), 3))

    for window in ROLLING_WINDOWS:
        print(
            f"Last {window} observations' median:",
            round(float(np.median(train[-window:])), 3),
        )

    print("Test-set mean (descriptive only):", round(float(np.mean(test)), 3))


if __name__ == "__main__":
    main()
