import pandas as pd
import numpy as np

FILE = "data/Final_data.csv"

df = pd.read_csv(FILE)

# Keep Maharashtra and the exact Dengue label.
df = df[
    df["state_ut"].astype(str).str.strip().str.lower().eq("maharashtra")
    & df["Disease"].astype(str).str.strip().eq("Dengue")
].copy()

df["report_date"] = pd.to_datetime(
    {
        "year": pd.to_numeric(df["year"], errors="coerce"),
        "month": pd.to_numeric(df["mon"], errors="coerce"),
        "day": pd.to_numeric(df["day"], errors="coerce"),
    },
    errors="coerce",
)

df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["report_date", "Cases", "district"])
df["district"] = df["district"].astype(str).str.strip()
df["month"] = df["report_date"].dt.month

# One observation per district-date, using the same median
# aggregation rule as the earlier experiment.
df = (
    df.groupby(["district", "report_date"], as_index=False)
    .agg(Cases=("Cases", "median"))
)
df["month"] = df["report_date"].dt.month
df = df.sort_values("report_date").reset_index(drop=True)

# Chronological split: first 80% of distinct dates for training.
dates = np.sort(df["report_date"].unique())
cutoff = dates[int(len(dates) * 0.8)]

train = df[df["report_date"] < cutoff].copy()
test = df[df["report_date"] >= cutoff].copy()

print("Training rows:", len(train))
print("Test rows:", len(test))
print("Training date range:",
      train["report_date"].min().date(), "to",
      train["report_date"].max().date())
print("Test date range:",
      test["report_date"].min().date(), "to",
      test["report_date"].max().date())
print("Test cutoff:", pd.Timestamp(cutoff).date())

global_median = train["Cases"].median()

district_median = train.groupby("district")["Cases"].median()
district_month_median = train.groupby(
    ["district", "month"]
)["Cases"].median()
month_median = train.groupby("month")["Cases"].median()

predictions = test[
    ["district", "report_date", "month", "Cases"]
].copy()
predictions = predictions.rename(columns={"Cases": "actual_cases"})

predictions["global_median"] = global_median

predictions["district_median"] = (
    predictions["district"].map(district_median)
    .fillna(global_median)
)

predictions["district_month_median"] = [
    district_month_median.get((d, m), district_median.get(d, global_median))
    for d, m in zip(predictions["district"], predictions["month"])
]

predictions["month_median"] = (
    predictions["month"].map(month_median).fillna(global_median)
)

models = [
    "global_median",
    "district_median",
    "district_month_median",
    "month_median",
]

results = []
for model in models:
    errors = predictions[model] - predictions["actual_cases"]
    results.append({
        "model": model,
        "MAE": errors.abs().mean(),
        "RMSE": np.sqrt((errors ** 2).mean()),
    })

results = pd.DataFrame(results).sort_values("MAE")

print("\n=== BASELINE COMPARISON ===")
print(results.round(3).to_string(index=False))

print("\n=== TEST SET DISTRICT COVERAGE ===")
print("Districts:", test["district"].nunique())
print("Districts unseen in training:",
      (~test["district"].isin(train["district"])).sum(),
      "test observations")

predictions.to_csv(
    "research_outputs/mh_dengue_baseline_predictions.csv",
    index=False,
)
results.to_csv(
    "research_outputs/mh_dengue_baseline_comparison.csv",
    index=False,
)

print("\nSaved baseline predictions and comparison in research_outputs/")
