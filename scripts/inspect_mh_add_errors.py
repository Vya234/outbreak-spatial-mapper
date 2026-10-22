import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = Path("research_outputs")
pred = pd.read_csv(OUT / "mh_add_baseline_predictions.csv")
pred["date"] = pd.to_datetime(pred["date"])

models = {
    "global_median": "global_median_pred",
    "district_median": "district_median_pred",
    "month_median": "month_median_pred",
}

print("=== TEST CASE DISTRIBUTION ===")
print(pred["Cases"].describe(
    percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
).round(2).to_string())

print("\n=== LARGEST RECORDED TEST COUNTS ===")
cols = ["date", "district", "Cases"] + list(models.values())
print(pred.nlargest(10, "Cases")[cols].to_string(index=False))

print("\n=== ERRORS BY TEST YEAR ===")
pred["year"] = pred["date"].dt.year
for year, g in pred.groupby("year"):
    print(f"\n{year}: n={len(g)}")
    for model, col in models.items():
        mae = mean_absolute_error(g["Cases"], g[col])
        rmse = np.sqrt(mean_squared_error(g["Cases"], g[col]))
        print(f"  {model}: MAE={mae:.2f}, RMSE={rmse:.2f}")

print("\n=== TEST DISTRICT COVERAGE BY YEAR ===")
print(
    pred.groupby("year").agg(
        observations=("Cases", "size"),
        districts=("district", "nunique"),
        median_cases=("Cases", "median"),
        max_cases=("Cases", "max"),
    ).round(2).to_string()
)
