import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_engineering import compute_daily_features

DATA_PATH = ROOT / "data" / "processed_dengue_mh.csv"
OUTPUT_DIR = ROOT / "research_outputs"

MIN_TRAIN_SIZE = 120
FORECAST_HORIZON = 7


def load_daily_counts():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    features = compute_daily_features(df, cumulative=True)

    counts = pd.to_numeric(
        features["newly_logged"], errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Daily counts contain invalid values.")

    return features.reset_index(drop=True), counts


def generate_forecasts(history):
    global_median = float(np.median(history))
    recent_median = float(np.median(history[-7:]))

    ensemble = (global_median + recent_median) / 2

    return {
        "Expanding Training Median": global_median,
        "Recent 7-Day Median": recent_median,
        "50-50 Median Ensemble": ensemble,
    }


def run_validation(features, counts):
    detailed_rows = []
    block_rows = []

    origin = MIN_TRAIN_SIZE
    block = 1

    while origin < len(counts):
        history = counts[:origin]
        end = min(origin + FORECAST_HORIZON, len(counts))
        actual = counts[origin:end]

        for model, prediction in generate_forecasts(history).items():
            predicted = np.full(len(actual), prediction)

            block_rows.append({
                "block": block,
                "origin_index": origin,
                "training_days": len(history),
                "forecast_days": len(actual),
                "model": model,
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            })

            for offset, (observed, estimate) in enumerate(
                zip(actual, predicted)
            ):
                index = origin + offset

                row = {
                    "block": block,
                    "day_index": index,
                    "model": model,
                    "actual_logged_records": float(observed),
                    "predicted_logged_records": float(estimate),
                    "absolute_error": float(abs(observed - estimate)),
                    "squared_error": float((observed - estimate) ** 2),
                }

                if "day" in features.columns:
                    row["timeline_day"] = features.iloc[index]["day"]

                detailed_rows.append(row)

        origin += FORECAST_HORIZON
        block += 1

    return pd.DataFrame(detailed_rows), pd.DataFrame(block_rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features, counts = load_daily_counts()
    detailed, blocks = run_validation(features, counts)

    summaries = []

    for model, group in detailed.groupby("model", sort=False):
        actual = group["actual_logged_records"].to_numpy()
        predicted = group["predicted_logged_records"].to_numpy()
        model_blocks = blocks[blocks["model"] == model]

        summaries.append({
            "model": model,
            "forecast_blocks": model_blocks["block"].nunique(),
            "total_test_days": len(group),
            "pooled_MAE": mean_absolute_error(actual, predicted),
            "pooled_RMSE": np.sqrt(
                mean_squared_error(actual, predicted)
            ),
            "mean_block_MAE": model_blocks["MAE"].mean(),
            "mean_block_RMSE": model_blocks["RMSE"].mean(),
        })

    summary = pd.DataFrame(summaries).sort_values(
        ["pooled_MAE", "pooled_RMSE"]
    ).reset_index(drop=True)

    detailed.to_csv(OUTPUT_DIR / "7day_ensemble_detailed.csv", index=False)
    blocks.to_csv(OUTPUT_DIR / "7day_ensemble_block_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "7day_ensemble_summary.csv", index=False)

    print("=" * 65)
    print("7-DAY ENSEMBLE VALIDATION")
    print("=" * 65)
    print(f"Total observations: {len(counts)}")
    print(f"Forecast horizon: {FORECAST_HORIZON} days")
    print("\nSUMMARY — LOWER ERRORS ARE BETTER\n")
    print(summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}"
    ))

    print("\nFiles saved in:", OUTPUT_DIR)
    print(" - 7day_ensemble_detailed.csv")
    print(" - 7day_ensemble_block_metrics.csv")
    print(" - 7day_ensemble_summary.csv")
    print("\nTarget: daily logged-coordinate record counts,")
    print("not necessarily actual new infections.")


if __name__ == "__main__":
    main()
