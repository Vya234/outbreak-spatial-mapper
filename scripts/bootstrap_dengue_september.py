from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("research_outputs")
comparison = pd.read_csv(
    OUT / "mh_dengue_september_district_comparison.csv"
)

col = "difference_september_minus_other_months"
x = comparison[col].dropna().to_numpy(dtype=float)

if len(x) == 0:
    raise ValueError("No district comparisons available.")

rng = np.random.default_rng(42)
B = 20000

boot = np.array([
    np.median(rng.choice(x, size=len(x), replace=True))
    for _ in range(B)
])

observed_median = float(np.median(x))
ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

print("Districts included:", len(x))
print("Median district-level September difference:", observed_median)
print("Bootstrap 95% CI:", (float(ci_low), float(ci_high)))
print("Positive differences:", int((x > 0).sum()))
print("Zero differences:", int((x == 0).sum()))
print("Negative differences:", int((x < 0).sum()))

result = pd.DataFrame([{
    "districts_included": len(x),
    "median_difference": observed_median,
    "bootstrap_ci_2_5": ci_low,
    "bootstrap_ci_97_5": ci_high,
    "positive_districts": int((x > 0).sum()),
    "equal_districts": int((x == 0).sum()),
    "negative_districts": int((x < 0).sum()),
    "seed": 42,
    "bootstrap_replicates": B,
}])

result.to_csv(OUT / "mh_dengue_september_bootstrap.csv", index=False)
print("\nSaved research_outputs/mh_dengue_september_bootstrap.csv")
