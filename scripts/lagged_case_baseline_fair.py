
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = Path("research_outputs")
INPUT = OUT / "mh_add_district_date_standardized.csv"
OUTPUT = OUT / "lagged_case_baseline_results_fair.csv"

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")

df = df.dropna(subset=["date", "district", "Cases"]).copy()
df["district"] = df["district"].astype(str).str.strip()
df = df[df["district"] != ""].copy()
df = df[df["Cases"] >= 0].copy()

# This experiment assumes one row per district per date.
# Stop rather than silently treating same-date records as separate lags.
duplicates = df.duplicated(["district", "date"], keep=False)
if duplicates.any():
    print(df.loc[duplicates, ["district", "date", "Cases"]]
            .sort_values(["district", "date"])
            .head(20).to_string(index=False))
    raise ValueError(
        "Multiple rows exist for a district/date. Resolve aggregation "
        "semantics before creating lag features."
    )

df["month"] = df["date"].dt.month
df = df.sort_values(["district", "date"]).reset_index(drop=True)

# Previous reported case count in the same district.
# No missing dates are filled with zero.
df["lag_cases"] = df.groupby("district")["Cases"].shift(1)

dates = np.sort(df["date"].unique())
results = []

for fraction in [0.60, 0.70, 0.80]:
    cutoff = pd.Timestamp(dates[int(len(dates) * fraction)])

    train_all = df[df["date"] < cutoff].copy()
    test_all = df[df["date"] >= cutoff].copy()

    # Use a shared test set for every model: test rows with a known
    # previous district observation. This makes the comparison fair.
    train_lag = train_all.dropna(subset=["lag_cases"]).copy()
    test = test_all.dropna(subset=["lag_cases"]).copy()

    if len(train_all) < 30 or len(train_lag) < 30 or len(test) < 10:
        print("Skipping", cutoff.date(), "- insufficient rows")
        continue

    y_test = test["Cases"].to_numpy()
    baseline_median = train_lag["Cases"].median()

    predictions = {
        "Training median": np.full(len(test), baseline_median),
        "Previous district observation": test["lag_cases"].to_numpy(),
    }

    # All models use the same test rows and the same eligible training
    # population for a fair comparison against the median baseline.
    features = ["district", "month", "lag_cases"]

    prep = ColumnTransformer([
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            ["district", "month"],
        ),
        (
            "num",
            SimpleImputer(strategy="median"),
            ["lag_cases"],
        ),
    ])

    model = make_pipeline(
        prep,
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    )

    model.fit(train_lag[features], train_lag["Cases"])
    predictions["District + month + lagged cases"] = np.maximum(
        0, model.predict(test[features])
    )

    rows = []
    for name, pred in predictions.items():
        rows.append({
            "cutoff": cutoff.date(),
            "model": name,
            "train_n": len(train_lag),
            "test_n": len(test),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        })

    base = next(r for r in rows if r["model"] == "Training median")

    for r in rows:
        r["MAE_improvement_pct"] = (
            100 * (base["MAE"] - r["MAE"]) / base["MAE"]
            if base["MAE"] != 0 else np.nan
        )
        r["RMSE_improvement_pct"] = (
            100 * (base["RMSE"] - r["RMSE"]) / base["RMSE"]
            if base["RMSE"] != 0 else np.nan
        )

    results.extend(rows)

if not results:
    raise RuntimeError("No evaluation results were produced.")

out = pd.DataFrame(results)
print(out.round(3).to_string(index=False))
out.to_csv(OUTPUT, index=False)

print(f"\nSaved {OUTPUT}")
print("Evaluation: chronological, rolling one-step-ahead predictions.")
print("Test rows are shared across all three models.")
print("Previous reported observations are used; missing dates are not zero-filled.")
print("This is not a fixed multi-day forecast.")
