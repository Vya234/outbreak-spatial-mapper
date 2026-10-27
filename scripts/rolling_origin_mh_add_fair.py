
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
df["district"] = df["district"].astype(str)
df["month"] = df["date"].dt.month

df = (
    df.sort_values("date")
      .drop_duplicates()
      .reset_index(drop=True)
)

dates = np.array(sorted(df["date"].unique()))
results = []

configs = [
    ("Training median", [], []),
    ("District + month", ["district", "month"], []),
    (
        "District + month + climate",
        ["district", "month"],
        ["Temp", "preci", "LAI"],
    ),
]

for fraction in [0.60, 0.70, 0.80]:
    cutoff = pd.Timestamp(dates[int(len(dates) * fraction)])

    # Chronological split only: do not filter by climate availability.
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()

    if len(train) < 30 or len(test) < 10:
        print("Skipping", cutoff.date(), "- insufficient rows")
        continue

    y_test = test["Cases"].to_numpy()
    cutoff_rows = []

    for name, categorical, numeric in configs:
        # Each model uses every training row eligible for its features.
        if name == "Training median":
            eligible_train = train.dropna(subset=["Cases"])
            pred = np.full(
                len(test),
                eligible_train["Cases"].median(),
            )
            train_n = len(eligible_train)

        else:
            features = categorical + numeric

            # Climate missingness is handled inside the pipeline.
            transformers = [
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical,
                )
            ]

            if numeric:
                transformers.append(
                    (
                        "num",
                        SimpleImputer(strategy="median"),
                        numeric,
                    )
                )

            prep = ColumnTransformer(transformers)

            model = make_pipeline(
                prep,
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            )

            # No climate-based row deletion.
            model.fit(train[features], train["Cases"])
            pred = np.maximum(
                0,
                model.predict(test[features]),
            )
            train_n = len(train)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        cutoff_rows.append({
            "cutoff": cutoff.date(),
            "train_n": train_n,
            "test_n": len(test),
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
        })

    base = next(
        r for r in cutoff_rows
        if r["model"] == "Training median"
    )

    for r in cutoff_rows:
        r["MAE_improvement_pct"] = (
            100 * (base["MAE"] - r["MAE"]) / base["MAE"]
            if base["MAE"] != 0 else np.nan
        )
        r["RMSE_improvement_pct"] = (
            100 * (base["RMSE"] - r["RMSE"]) / base["RMSE"]
            if base["RMSE"] != 0 else np.nan
        )

    results.extend(cutoff_rows)

out = pd.DataFrame(results)

if out.empty:
    raise RuntimeError("No evaluation results were produced.")

print(out.round(3).to_string(index=False))

output_path = OUT / "rolling_origin_mh_add_fair_results_corrected.csv"
out.to_csv(output_path, index=False)

print(f"\nSaved {output_path}")
print("All models use the same chronological test rows.")
print("Training rows are not filtered by climate availability.")
print("Climate features are imputed using training-set statistics.")
print("Caution: target-date climate is used; this is retrospective")
print("case-count evaluation, not a deployable future forecast.")
