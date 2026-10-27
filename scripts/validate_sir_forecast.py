import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Add the project root to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_engineering import compute_daily_features
from forecasting_engine import optimize_sir_parameters, sir_model_ode


DATA_PATH = PROJECT_ROOT / "data" / "processed_dengue_mh.csv"
OUTPUT_PATH = PROJECT_ROOT / "research_outputs" / "sir_holdout_validation.csv"

TRAIN_FRACTION = 0.70


def calculate_metrics(actual, predicted):
    """Calculate MAE and RMSE for actual and predicted values."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError(
            f"Metric length mismatch: actual={len(actual)}, "
            f"predicted={len(predicted)}"
        )

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return float(mae), float(rmse)


def main():
    # ---------------------------------------------------------
    # 1. Load and prepare the dataset
    # ---------------------------------------------------------
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

    if len(train) < 3 or len(test) < 1:
        raise ValueError("Not enough observations for train/test evaluation.")

    print("Total daily observations:", len(counts))
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

    # ---------------------------------------------------------
    # 2. Median baseline
    # ---------------------------------------------------------
    # Predict the median daily count observed in the training set
    # for every day in the held-out test period.
    baseline_prediction = np.full(len(test), np.median(train))

    baseline_mae, baseline_rmse = calculate_metrics(
        test,
        baseline_prediction,
    )

    print("\nMEDIAN BASELINE")
    print("MAE:", round(baseline_mae, 3))
    print("RMSE:", round(baseline_rmse, 3))

    # ---------------------------------------------------------
    # 3. Fit SIR parameters using training data only
    # ---------------------------------------------------------
    fit = optimize_sir_parameters(
        train,
        initial_population=50000,
    )

    print("\nSIR FIT")
    print(fit)

    if not fit["success"]:
        raise RuntimeError("SIR optimization did not converge.")

    N = float(fit["optimized_N"])
    beta = float(fit["optimized_beta"])
    gamma = float(fit["optimized_gamma"])

    initial_infected = float(train[0])

    if N <= initial_infected:
        raise ValueError("Fitted population is too small.")

    # ---------------------------------------------------------
    # 4. Simulate through the end of the test period
    # ---------------------------------------------------------
    total_observations = len(train) + len(test)
    t = np.arange(total_observations, dtype=float)

    S0 = N - initial_infected
    I0 = initial_infected
    R0 = 0.0

    solution = odeint(
        sir_model_ode,
        [S0, I0, R0],
        t,
        args=(N, beta, gamma),
    )

    # I + R represents cumulative infections in this SIR formulation.
    cumulative_prediction = solution[:, 1] + solution[:, 2]

    # Convert the cumulative trajectory to daily increments.
    # diff[k] represents the predicted increment from time k to k + 1.
    daily_prediction = np.diff(cumulative_prediction)

    # Avoid negative predicted daily counts caused by numerical effects.
    daily_prediction = np.maximum(daily_prediction, 0.0)

    # ---------------------------------------------------------
    # 5. Extract predictions for the held-out test period
    # ---------------------------------------------------------
    # The first test observation is at index len(train).
    # Its predicted increment is the change from the last training
    # time point to the first test time point, at index len(train)-1
    # in the differenced array.
    test_prediction = daily_prediction[len(train) - 1:]

    if len(test_prediction) != len(test):
        raise RuntimeError(
            "Prediction and test lengths do not match: "
            f"{len(test_prediction)} predictions versus "
            f"{len(test)} test observations."
        )

    sir_mae, sir_rmse = calculate_metrics(
        test,
        test_prediction,
    )

    print("\nSIR DAILY-COUNT FORECAST")
    print("MAE:", round(sir_mae, 3))
    print("RMSE:", round(sir_rmse, 3))

    # ---------------------------------------------------------
    # 6. Compare the models
    # ---------------------------------------------------------
    print("\nCOMPARISON")
    print("MAE difference (SIR - baseline):", round(sir_mae - baseline_mae, 3))
    print(
        "RMSE difference (SIR - baseline):",
        round(sir_rmse - baseline_rmse, 3),
    )

    if sir_mae < baseline_mae:
        print("SIR has lower MAE on this held-out period.")
    elif sir_mae > baseline_mae:
        print("The median baseline has lower MAE.")
    else:
        print("Both models have equal MAE.")

    if sir_rmse < baseline_rmse:
        print("SIR has lower RMSE on this held-out period.")
    elif sir_rmse > baseline_rmse:
        print("The median baseline has lower RMSE.")
    else:
        print("Both models have equal RMSE.")

    # ---------------------------------------------------------
    # 7. Save held-out predictions
    # ---------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({
        "day": daily["day"].iloc[split:].to_numpy(),
        "actual_daily_logged": test,
        "median_baseline_prediction": baseline_prediction,
        "sir_daily_prediction": test_prediction,
    })

    results.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved held-out predictions to:", OUTPUT_PATH)
    print("Saved prediction rows:", len(results))


if __name__ == "__main__":
    main()
