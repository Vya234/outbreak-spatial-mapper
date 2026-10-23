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
df = pd.read_csv(OUT / "mh_add_district_date.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

for c in ["Cases", "Temp", "preci", "LAI"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["date", "district", "Cases"]).copy()
df["month"] = df["date"].dt.month
df = df.sort_values("date")

dates = np.array(sorted(df["date"].unique()))
cutoff = pd.Timestamp(dates[int(len(dates) * 0.8)])

train = df[df["date"] < cutoff].copy()
test = df[df["date"] >= cutoff].dropna(
    subset=["Temp", "preci", "LAI"]
).copy()
train = train.dropna(subset=["Temp", "preci", "LAI"]).copy()

print("Cutoff:", cutoff.date())
print("Climate-complete train:", len(train))
print("Common test rows:", len(test))

if len(train) < 30 or len(test) < 10:
    raise SystemExit("Insufficient rows; no results generated.")

y_train = train["Cases"]
y_test = test["Cases"]
baseline = np.full(len(test), y_train.median())

configs = [
    ("Training median", None, []),
    ("District + month", ["district", "month"], []),
    ("District + month + climate",
     ["district", "month"], ["Temp", "preci", "LAI"]),
]
results = []

for name, categorical, numeric in configs:
    if categorical is None:
        pred = baseline
    else:
        features = categorical + numeric
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", SimpleImputer(strategy="median"), numeric),
        ])
        model = make_pipeline(
            prep,
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            )
        )
        model.fit(train[features], y_train)
        pred = np.maximum(0, model.predict(test[features]))

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    results.append({
        "model": name, "cutoff": cutoff.date(),
        "train_n": len(train), "test_n": len(test),
        "MAE": mae, "RMSE": rmse,
        "MAE_change_vs_baseline_pct":
            100 * (results[0]["MAE"] - mae) / results[0]["MAE"]
            if results else 0,
        "RMSE_change_vs_baseline_pct":
            100 * (results[0]["RMSE"] - rmse) / results[0]["RMSE"]
            if results else 0,
    })

summary = pd.DataFrame(results)
print("\nSAME TEST ROWS — RETROSPECTIVE CASE-COUNT EVALUATION")
print(summary.round(3).to_string(index=False))
summary.to_csv(OUT / "mh_add_fair_model_comparison.csv", index=False)
print("\nSaved research_outputs/mh_add_fair_model_comparison.csv")
print("Caution: same-date climate measurements are not necessarily")
print("available in advance; this is not an operational forecast.")
