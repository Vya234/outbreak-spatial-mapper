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

dedup_cols = [c for c in df.columns if c != "Unnamed: 0"]
df = df.drop_duplicates(subset=dedup_cols).copy()
df["Month"] = df["Date"].dt.month

dates = np.sort(df["Date"].unique())
cutoff = pd.Timestamp(dates[int(len(dates) * 0.8)])

train = df[df["Date"] < cutoff].copy()
test = df[df["Date"] >= cutoff].copy()

y_train = train["Cases"].astype(float)
y_test = test["Cases"].astype(float)

def build_model(numeric_features):
    prep = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median", add_indicator=True),
            numeric_features,
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

    return Pipeline([
        ("prep", prep),
        ("rf", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ])

experiments = {
    "District + month": ["Month"],
    "District + month + climate": ["Month", "Temp", "preci", "LAI"],
}

rows = [{
    "Model": "Training median",
    "MAE": mean_absolute_error(
        y_test, np.full(len(test), y_train.median())
    ),
    "RMSE": np.sqrt(mean_squared_error(
        y_test, np.full(len(test), y_train.median())
    )),
}]

for name, numeric_features in experiments.items():
    model = build_model(numeric_features)
    model.fit(train[["district"] + numeric_features], y_train)

    pred = np.maximum(
        0, model.predict(test[["district"] + numeric_features])
    )

    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    })

results = pd.DataFrame(rows)
results.to_csv(OUT / "climate_model_comparison.csv", index=False)

print("Cutoff:", cutoff.date())
print("Train rows:", len(train), "| Test rows:", len(test))
print(results.round(3).to_string(index=False))
print("\nSaved:", OUT / "climate_model_comparison.csv")
