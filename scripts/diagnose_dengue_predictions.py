import pandas as pd
import numpy as np

p = pd.read_csv("research_outputs/mh_dengue_test_predictions.csv")

models = {
    "Median baseline": "training_median_baseline_prediction",
    "District RF": "a_district_prediction",
    "District + month RF": "b_district_month_prediction",
    "District + month + climate RF": "c_district_month_climate_prediction",
}

# Overall metrics
print("\n=== OVERALL TEST PERFORMANCE ===")
for name, col in models.items():
    err = p[col] - p["actual_cases"]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    print(f"{name:32s} MAE={mae:8.3f} RMSE={rmse:8.3f}")

# Performance on ordinary vs high-count observations
p["case_group"] = pd.cut(
    p["actual_cases"],
    bins=[-np.inf, 10, 25, np.inf],
    labels=["Low (<=10)", "Medium (11-25)", "High (>25)"]
)

print("\n=== ERROR BY ACTUAL CASE COUNT ===")
for group, g in p.groupby("case_group", observed=True):
    print(f"\n{group}: n={len(g)}")
    for name, col in models.items():
        print(f"  {name:30s} MAE={(g[col]-g['actual_cases']).abs().mean():.3f}")

# Best and worst RF predictions
rf_col = "a_district_prediction"
p["rf_abs_error"] = (p[rf_col] - p["actual_cases"]).abs()

print("\n=== 10 LARGEST DISTRICT-ONLY RF ERRORS ===")
print(
    p.sort_values("rf_abs_error", ascending=False)[
        ["district", "report_date", "actual_cases", rf_col, "rf_abs_error"]
    ].head(10).round(2).to_string(index=False)
)

# Bias by district; interpret cautiously for districts with few records
print("\n=== DISTRICT-LEVEL DISTRICT-ONLY RF PERFORMANCE ===")
rows = []
for district, g in p.groupby("district"):
    rows.append({
        "district": district,
        "n": len(g),
        "MAE": (g[rf_col] - g["actual_cases"]).abs().mean(),
        "mean_error": (g[rf_col] - g["actual_cases"]).mean()
    })

print(
    pd.DataFrame(rows)
    .sort_values("n", ascending=False)
    .round(2)
    .to_string(index=False)
)

# Compare predictions for the largest case counts
print("\n=== PREDICTIONS FOR OBSERVATIONS WITH MORE THAN 25 CASES ===")
cols = ["district", "report_date", "actual_cases"] + list(models.values())
print(
    p[p["actual_cases"] > 25][cols]
    .sort_values("actual_cases", ascending=False)
    .round(2)
    .to_string(index=False)
)
