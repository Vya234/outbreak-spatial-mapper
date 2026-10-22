from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs"
OUT.mkdir(exist_ok=True)

# Load the original source.
df = pd.read_csv(ROOT / "data" / "Final_data.csv")

df = df[
    df["state_ut"].astype(str).str.strip().eq("Maharashtra")
    & df["Disease"].astype(str).str.strip().eq(
        "Acute Diarrhoeal Disease"
    )
].copy()

df["Date"] = pd.to_datetime(
    dict(
        year=pd.to_numeric(df["year"], errors="coerce"),
        month=pd.to_numeric(df["mon"], errors="coerce"),
        day=pd.to_numeric(df["day"], errors="coerce"),
    ),
    errors="coerce",
)

for col in ["Cases", "Temp", "preci", "LAI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Date", "Cases"]).copy()

# Deduplicate exact repeated observations in the modelling copy only.
# Keep the source data and audit CSV unchanged.
duplicate_columns = [
    col for col in df.columns
    if col not in ["Unnamed: 0"]
]
before = len(df)
df = df.drop_duplicates(subset=duplicate_columns).copy()
removed = before - len(df)

# Simple calendar feature. Month is available from the recorded date.
df["Month"] = df["Date"].dt.month

# Hold out the latest 20% of distinct dates.
unique_dates = np.sort(df["Date"].unique())
cut_index = int(len(unique_dates) * 0.8)
cut_index = min(max(cut_index, 1), len(unique_dates) - 1)
cutoff = pd.Timestamp(unique_dates[cut_index])

train = df[df["Date"] < cutoff].copy()
test = df[df["Date"] >= cutoff].copy()

categorical = ["district"]
numeric = ["Month", "Temp", "preci", "LAI"]
features = categorical + numeric

X_train = train[features]
y_train = train["Cases"].astype(float)
X_test = test[features]
y_test = test["Cases"].astype(float)

# Baseline: training median.
baseline_pred = np.full(len(test), y_train.median())

# Preprocessing is fitted only on training data.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median", add_indicator=True),
            numeric,
        ),
        (
            "district",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

model.fit(X_train, y_train)
model_pred = np.maximum(0, model.predict(X_test))

def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    }

results = pd.DataFrame([
    {"Model": "Training-median baseline",
     **metrics(y_test, baseline_pred)},
    {"Model": "Pooled Random Forest",
     **metrics(y_test, model_pred)},
])

print("=== DATASET ===")
print("Rows after exact deduplication:", len(df))
print("Exact duplicate rows removed from model copy:", removed)
print("Districts:", df["district"].nunique())
print("Training rows:", len(train))
print("Test rows:", len(test))
print("Train end:", train["Date"].max())
print("Test start:", test["Date"].min())
print("Test end:", test["Date"].max())
print("\n=== TEST METRICS ===")
print(results.round(3).to_string(index=False))

predictions = test[
    ["Date", "district", "Cases", "Month", "Temp", "preci", "LAI"]
].copy()
predictions["Baseline_Prediction"] = baseline_pred
predictions["RandomForest_Prediction"] = model_pred
predictions.to_csv(OUT / "pooled_model_test_predictions.csv", index=False)
results.to_csv(OUT / "pooled_model_metrics.csv", index=False)

print("\nSaved:")
print(OUT / "pooled_model_test_predictions.csv")
print(OUT / "pooled_model_metrics.csv")

print(
    "\nInterpretation: this predicts recorded case counts for later "
    "observations. It is an initial benchmark, not yet a validated "
    "weekly outbreak forecast."
)
