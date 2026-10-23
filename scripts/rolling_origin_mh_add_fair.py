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
df = df.sort_values("date").drop_duplicates().reset_index(drop=True)

dates = np.array(sorted(df["date"].unique()))
results = []

for fraction in [0.60, 0.70, 0.80]:
    cutoff = pd.Timestamp(dates[int(len(dates) * fraction)])
    train = df[df["date"] < cutoff].dropna(
        subset=["Temp", "preci", "LAI"]
    ).copy()
    test = df[df["date"] >= cutoff].dropna(
        subset=["Temp", "preci", "LAI"]
    ).copy()

    if len(train) < 30 or len(test) < 10:
        print("Skipping", cutoff.date(), "- insufficient rows")
        continue

    y_train = train["Cases"]
    y_test = test["Cases"]
    baseline = np.full(len(test), y_train.median())

    configs = [
        ("Training median", None, []),
        ("District + month", ["district", "month"], []),
        ("District + month + climate",
         ["district", "month"], ["Temp", "preci", "LAI"]),
    ]

    cutoff_rows = []
    for name, categorical, numeric in configs:
        if categorical is None:
            pred = baseline
        else:
            features = categorical + numeric
            prep = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"),
                 categorical),
                ("num", SimpleImputer(strategy="median"), numeric),
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
            model.fit(train[features], y_train)
            pred = np.maximum(0, model.predict(test[features]))

        cutoff_rows.append({
            "cutoff": cutoff.date(),
            "model": name,
            "train_n": len(train),
            "test_n": len(test),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        })

    base = next(r for r in cutoff_rows if r["model"] == "Training median")
    for r in cutoff_rows:
        r["MAE_improvement_pct"] = (
            100 * (base["MAE"] - r["MAE"]) / base["MAE"]
        )
        r["RMSE_improvement_pct"] = (
            100 * (base["RMSE"] - r["RMSE"]) / base["RMSE"]
        )
    results.extend(cutoff_rows)

out = pd.DataFrame(results)
print(out.round(3).to_string(index=False))
out.to_csv(OUT / "rolling_origin_mh_add_fair_results.csv", index=False)
print("\nSaved research_outputs/rolling_origin_mh_add_fair_results.csv")
print("Caution: climate values are from the target dates.")
print("This is retrospective case-count evaluation, not a future forecast.")
