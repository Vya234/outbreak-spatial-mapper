from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = Path("research_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Load and filter the exact-label Maharashtra Dengue subset
df = pd.read_csv("data/Final_data.csv")

for col in ["state_ut", "Disease", "district"]:
    df[col] = df[col].astype("string").str.strip()

d = df[
    (df["state_ut"] == "Maharashtra")
    & (df["Disease"] == "Dengue")
].copy()

for col in ["day", "mon", "year", "Cases", "Temp", "preci", "LAI"]:
    d[col] = pd.to_numeric(d[col], errors="coerce")

d["report_date"] = pd.to_datetime(
    {"year": d["year"], "month": d["mon"], "day": d["day"]},
    errors="coerce",
)

d = d.dropna(subset=["district", "report_date", "Cases"]).copy()
d["month"] = d["report_date"].dt.month

# Aggregate repeated district-date records for this sensitivity analysis.
# Median is used because some repeated records have conflicting case counts.
agg = (
    d.groupby(["district", "report_date"], as_index=False)
    .agg(
        Cases=("Cases", "median"),
        month=("month", "first"),
        Temp=("Temp", "median"),
        preci=("preci", "median"),
        LAI=("LAI", "median"),
    )
    .sort_values("report_date")
    .reset_index(drop=True)
)

# Chronological split by distinct date, so records on the same date
# cannot appear on both sides of the split.
dates = np.sort(agg["report_date"].unique())
if len(dates) < 5:
    raise ValueError("Too few distinct reporting dates for a time split.")

cut = int(len(dates) * 0.8)
cut = min(max(cut, 1), len(dates) - 1)
cutoff_date = pd.Timestamp(dates[cut])

train = agg[agg["report_date"] < cutoff_date].copy()
test = agg[agg["report_date"] >= cutoff_date].copy()

print("=== DATA SPLIT ===")
print("District-date rows:", len(agg))
print("Distinct dates:", len(dates))
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Test begins:", cutoff_date.date())
print("Train date range:", train["report_date"].min().date(),
      "to", train["report_date"].max().date())
print("Test date range:", test["report_date"].min().date(),
      "to", test["report_date"].max().date())

if len(train) < 20 or len(test) < 10:
    print("WARNING: Small train/test sample; metrics may be unstable.")

# Baseline: predict the training-set median for every test row.
y_train = train["Cases"].to_numpy(dtype=float)
y_test = test["Cases"].to_numpy(dtype=float)

predictions = {}
predictions["Training median baseline"] = np.full(
    len(test), np.median(y_train)
)

# Model A: district only
# Model B: district + calendar month
# Model C: district + month + climate
feature_sets = {
    "A: District": ["district"],
    "B: District + month": ["district", "month"],
    "C: District + month + climate": [
        "district", "month", "Temp", "preci", "LAI"
    ],
}

for name, features in feature_sets.items():
    categorical = [
        col for col in features
        if col in ["district", "month"]
    ]
    numeric = [
        col for col in features
        if col in ["Temp", "preci", "LAI"]
    ]

    transformers = []
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median",
                                          add_indicator=True)),
            ]),
            numeric,
        ))

    preprocess = ColumnTransformer(transformers=transformers)

    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(train[features], y_train)
    predictions[name] = model.predict(test[features])

# Evaluate all models on the exact same test rows.
rows = []
for name, pred in predictions.items():
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    rows.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "test_rows": len(y_test),
    })

results = pd.DataFrame(rows).sort_values("MAE")
results.to_csv(OUT / "mh_dengue_model_comparison.csv", index=False)

print("\n=== HELD-OUT PERFORMANCE ===")
print(results.round(3).to_string(index=False))

# Save row-level predictions for error analysis
prediction_output = test[
    ["district", "report_date", "Cases", "month"]
].copy()
prediction_output = prediction_output.rename(
    columns={"Cases": "actual_cases"}
)

for name, pred in predictions.items():
    safe_name = (
        name.lower().replace(":", "").replace(" + ", "_")
        .replace(" ", "_")
    )
    prediction_output[safe_name + "_prediction"] = pred

prediction_output.to_csv(
    OUT / "mh_dengue_test_predictions.csv", index=False
)

# Plot MAE comparison
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(results["model"], results["MAE"])
ax.set_ylabel("Test MAE (lower is better)")
ax.set_title("Maharashtra Dengue: Chronological Holdout Comparison")
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "mh_dengue_model_comparison.png", dpi=200)
plt.close(fig)

print("\nSaved:")
print(OUT / "mh_dengue_model_comparison.csv")
print(OUT / "mh_dengue_test_predictions.csv")
print(OUT / "mh_dengue_model_comparison.png")
