import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

# Reconstruct the chronological split from the saved district-date table.
df = pd.read_csv(OUT / "mh_add_district_date.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["month"] = df["date"].dt.month
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")

for col in ["Temp", "preci", "LAI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date", "district", "Cases"]).sort_values("date")
dates = np.array(sorted(df["date"].unique()))
cutoff = pd.Timestamp(dates[int(len(dates) * 0.8)])

train = df[df["date"] < cutoff].copy()
test = df[df["date"] >= cutoff].copy()

features = ["Temp", "preci", "LAI", "month"]
train_climate = train.dropna(subset=["Temp", "preci", "LAI"]).copy()
test_climate = test.dropna(subset=["Temp", "preci", "LAI"]).copy()

print("=== CHRONOLOGICAL SPLIT ===")
print("Cutoff:", cutoff.date())
print("All training rows:", len(train))
print("All test rows:", len(test))
print("Climate-complete training rows:", len(train_climate))
print("Climate-complete test rows:", len(test_climate))

if len(train_climate) < 30 or len(test_climate) < 10:
    raise SystemExit(
        "Too few climate-complete observations for this comparison. "
        "Do not impute the target or force the experiment."
    )

X_train = train_climate[features]
y_train = train_climate["Cases"]
X_test = test_climate[features]
y_test = test_climate["Cases"]

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Fair baseline: same test rows, using only the climate-complete training targets.
baseline = np.full(len(test_climate), y_train.median())

results = []
for name, y_pred in [
    ("global_median_climate_complete", baseline),
    ("random_forest_climate", pred),
]:
    results.append({
        "model": name,
        "n_test": len(y_test),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    })

summary = pd.DataFrame(results)
print("\n=== SAME-ROWS MODEL COMPARISON ===")
print(summary.round(3).to_string(index=False))

test_climate["rf_climate_prediction"] = pred
test_climate["global_median_prediction"] = baseline
summary.to_csv(OUT / "mh_add_climate_model_comparison.csv", index=False)
test_climate.to_csv(OUT / "mh_add_climate_predictions.csv", index=False)

print("\nSaved:", OUT / "mh_add_climate_model_comparison.csv")
print("Saved:", OUT / "mh_add_climate_predictions.csv")
print("This tests recorded cases at observed dates, not future outbreak spread.")
