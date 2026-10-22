import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("research_outputs/mh_add_district_date.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["month"] = df["date"].dt.month
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["date", "district", "Cases"]).sort_values("date")

# Chronological split by distinct dates to prevent same-date leakage.
dates = np.array(sorted(df["date"].unique()))
cutoff_idx = int(len(dates) * 0.8)
cutoff_idx = min(max(cutoff_idx, 1), len(dates) - 1)
cutoff = pd.Timestamp(dates[cutoff_idx])

train = df[df["date"] < cutoff].copy()
test = df[df["date"] >= cutoff].copy()

if train.empty or test.empty:
    raise ValueError("Train/test split is empty; inspect the date coverage.")

global_median = train["Cases"].median()
district_medians = train.groupby("district")["Cases"].median()
month_medians = train.groupby("month")["Cases"].median()

test = test.copy()
test["global_median_pred"] = global_median
test["district_median_pred"] = (
    test["district"].map(district_medians).fillna(global_median)
)
test["month_median_pred"] = (
    test["month"].map(month_medians).fillna(global_median)
)

print("=== CHRONOLOGICAL SPLIT ===")
print("Train rows:", len(train))
print("Test rows:", len(test))
print("Training dates:", train["date"].min().date(), "to",
      train["date"].max().date())
print("Testing dates:", test["date"].min().date(), "to",
      test["date"].max().date())
print("Training districts:", train["district"].nunique())
print("Test districts:", test["district"].nunique())
print("Test districts unseen in training:",
      len(set(test["district"]) - set(train["district"])))
print("Global training median:", round(global_median, 2))

results = []
for name, col in [
    ("global_median", "global_median_pred"),
    ("district_median", "district_median_pred"),
    ("month_median", "month_median_pred"),
]:
    y_true = test["Cases"]
    y_pred = test[col]
    results.append({
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    })

summary = pd.DataFrame(results).sort_values("MAE")
print("\n=== BASELINE COMPARISON ===")
print(summary.round(3).to_string(index=False))

summary.to_csv(OUT / "mh_add_baseline_comparison.csv", index=False)
test.to_csv(OUT / "mh_add_baseline_predictions.csv", index=False)

print("\nSaved:", OUT / "mh_add_baseline_comparison.csv")
print("Saved:", OUT / "mh_add_baseline_predictions.csv")
print("Caution: this evaluates recorded case counts at observed dates,")
print("not forecasts of a complete daily or weekly disease series.")
