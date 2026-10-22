import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs"
OUT.mkdir(exist_ok=True)

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

# Remove exact duplicates in this modelling copy only.
dedup_cols = [c for c in df.columns if c != "Unnamed: 0"]
df = df.drop_duplicates(subset=dedup_cols).copy()

df["Month"] = df["Date"].dt.month
df = df.sort_values(["district", "Date", "week_of_outbreak"])

# Previous observation within each district.
# This is a previous recorded value, not necessarily a one-week lag.
df["Previous_Cases"] = df.groupby("district")["Cases"].shift(1)

# Keep the same global chronological split as previous experiments.
dates = np.sort(df["Date"].unique())
cutoff = pd.Timestamp(dates[int(len(dates) * 0.8)])

train = df[df["Date"] < cutoff].copy()
test = df[df["Date"] >= cutoff].copy()

# A lag must exist; rows without one are excluded from the lag-model test.
train_lag = train.dropna(subset=["Previous_Cases"]).copy()
test_lag = test.dropna(subset=["Previous_Cases"]).copy()

features = [
    "district", "Month", "Previous_Cases",
    "Temp", "preci", "LAI"
]
numeric = ["Month", "Previous_Cases", "Temp", "preci", "LAI"]

prep = ColumnTransformer([
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
        ["district"],
    ),
])

model = Pipeline([
    ("prep", prep),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(train_lag[features], train_lag["Cases"])
pred = np.maximum(0, model.predict(test_lag[features]))

# Fair comparison on exactly the same test rows.
median_pred = np.full(len(test_lag), train_lag["Cases"].median())

results = pd.DataFrame([
    {
        "Model": "Training median (same test rows)",
        "MAE": mean_absolute_error(test_lag["Cases"], median_pred),
        "RMSE": np.sqrt(mean_squared_error(test_lag["Cases"], median_pred)),
    },
    {
        "Model": "Random Forest + previous case",
        "MAE": mean_absolute_error(test_lag["Cases"], pred),
        "RMSE": np.sqrt(mean_squared_error(test_lag["Cases"], pred)),
    },
])

print("Cutoff:", cutoff.date())
print("Training rows with lag:", len(train_lag))
print("Test rows with lag:", len(test_lag))
print(results.round(3).to_string(index=False))

results.to_csv(OUT / "lag_model_comparison.csv", index=False)

predictions = test_lag[["Date", "district", "Cases", "Previous_Cases"]].copy()
predictions["Prediction"] = pred
predictions.to_csv(OUT / "lag_model_predictions.csv", index=False)

print("\nSaved lag-model metrics and predictions.")
print(
    "Note: Previous_Cases is the previous recorded observation in each "
    "district; the elapsed time between observations is not fixed."
)
