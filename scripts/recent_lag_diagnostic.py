
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
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["date", "district", "Cases"]).copy()
df["district"] = df["district"].astype(str).str.strip()
df = df[df["Cases"] >= 0].copy()

# Require one row per district/date; do not silently aggregate.
if df.duplicated(["district", "date"]).any():
    raise ValueError(
        "Repeated district/date rows found. Resolve their meaning first."
    )

df = df.sort_values(["district", "date"]).reset_index(drop=True)
df["month"] = df["date"].dt.month
df["lag_cases"] = df.groupby("district")["Cases"].shift(1)
df["previous_date"] = df.groupby("district")["date"].shift(1)
df["gap_days"] = (df["date"] - df["previous_date"]).dt.days

dates = np.sort(df["date"].unique())
results = []

for fraction in [0.60, 0.70, 0.80]:
    cutoff = pd.Timestamp(dates[int(len(dates) * fraction)])
    train_all = df[df["date"] < cutoff].copy()
    test_all = df[df["date"] >= cutoff].copy()

    # Both training and test lags must be recent.
    train = train_all[
        train_all["gap_days"].between(1, 30)
    ].dropna(subset=["lag_cases"]).copy()

    test = test_all[
        test_all["gap_days"].between(1, 30)
    ].dropna(subset=["lag_cases"]).copy()

    print(
        f"\nCutoff: {cutoff.date()} | "
        f"all training rows: {len(train_all)} | "
        f"recent-lag training rows: {len(train)} | "
        f"recent-lag test rows: {len(test)}"
    )

    if len(train) < 30 or len(test) < 10:
        print("Insufficient recent-lag rows; skipping this cutoff.")
        continue

    y_test = test["Cases"].to_numpy()

    # Fair comparison: both baselines use the same training population.
    median_pred = np.full(len(test), train["Cases"].median())
    previous_pred = test["lag_cases"].to_numpy()

    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["district", "month"]),
        ("num", SimpleImputer(strategy="median"), ["lag_cases"]),
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

    model.fit(train[["district", "month", "lag_cases"]], train["Cases"])
    rf_pred = np.maximum(
        0,
        model.predict(test[["district", "month", "lag_cases"]]),
    )

    preds = {
        "Recent-lag training median": median_pred,
        "Previous observation (gap <= 30 days)": previous_pred,
        "District + month + recent lag": rf_pred,
    }

    rows = []
    for name, pred in preds.items():
        rows.append({
            "cutoff": cutoff.date(),
            "model": name,
            "train_n": len(train),
            "test_n": len(test),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        })

    base = rows[0]
    for row in rows:
        row["MAE_improvement_pct"] = (
            100 * (base["MAE"] - row["MAE"]) / base["MAE"]
            if base["MAE"] else np.nan
        )
        row["RMSE_improvement_pct"] = (
            100 * (base["RMSE"] - row["RMSE"]) / base["RMSE"]
            if base["RMSE"] else np.nan
        )

    results.extend(rows)

if results:
    out = pd.DataFrame(results)
    print("\n", out.round(3).to_string(index=False))
    path = OUT / "recent_lag_diagnostic_results.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved {path}")
else:
    print(
        "\nNo cutoff had enough recent-lag observations. "
        "Do not loosen the threshold automatically; inspect data coverage first."
    )
