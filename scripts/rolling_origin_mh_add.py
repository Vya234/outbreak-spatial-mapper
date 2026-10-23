import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = Path("research_outputs")
df = pd.read_csv(OUT / "mh_add_district_date.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["month"] = df["date"].dt.month

for col in ["Cases", "Temp", "preci", "LAI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date", "Cases"]).sort_values("date")
df = df.drop_duplicates().reset_index(drop=True)

features = ["Temp", "preci", "LAI", "month"]
dates = np.array(sorted(df["date"].unique()))
splits = [0.60, 0.70, 0.80]

rows = []

for fraction in splits:
    cutoff = pd.Timestamp(dates[int(len(dates) * fraction)])
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()

    train = train.dropna(subset=features)
    test = test.dropna(subset=features)

    if len(train) < 30 or len(test) < 10:
        print(f"Skipping cutoff {cutoff.date()}: insufficient rows")
        continue

    X_train, y_train = train[features], train["Cases"]
    X_test, y_test = test[features], test["Cases"]

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
    pred = np.maximum(0, model.predict(X_test))
    baseline = np.full(len(test), y_train.median())

    for name, values in [
        ("Training median", baseline),
        ("Random Forest climate", pred),
    ]:
        rows.append({
            "cutoff": cutoff.date(),
            "model": name,
            "train_n": len(train),
            "test_n": len(test),
            "MAE": mean_absolute_error(y_test, values),
            "RMSE": np.sqrt(mean_squared_error(y_test, values)),
        })

results = pd.DataFrame(rows)
print(results.round(3).to_string(index=False))
results.to_csv(OUT / "rolling_origin_mh_add_results.csv", index=False)
print("\nSaved research_outputs/rolling_origin_mh_add_results.csv")
