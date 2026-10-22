import pandas as pd
import numpy as np
from scipy.stats import binomtest

FILE = "research_outputs/mh_dengue_september_district_year_comparison.csv"
df = pd.read_csv(FILE).dropna(subset=["district", "difference"]).copy()

# Keep each district's multiple years together during resampling.
districts = df["district"].unique()
rng = np.random.default_rng(42)
B = 20000

observed_median = df["difference"].median()
observed_positive = (df["difference"] > 0).mean()

boot_medians = []
boot_positive = []

for _ in range(B):
    sampled = rng.choice(districts, size=len(districts), replace=True)

    # Repeatedly sampled districts must contribute repeatedly.
    pieces = [
        df[df["district"] == district]["difference"].to_numpy()
        for district in sampled
    ]
    values = np.concatenate(pieces)

    boot_medians.append(np.median(values))
    boot_positive.append(np.mean(values > 0))

lo, hi = np.percentile(boot_medians, [2.5, 97.5])
plo, phi = np.percentile(boot_positive, [2.5, 97.5])

print("Districts represented:", len(districts))
print("District-year comparisons:", len(df))
print("Observed median difference:", observed_median)
print("Observed positive fraction:", round(observed_positive, 3))
print("Cluster bootstrap 95% interval for median:", (lo, hi))
print("Cluster bootstrap 95% interval for positive fraction:", (plo, phi))

print("\nDistrict-level summary:")
district_summary = df.groupby("district").agg(
    years=("difference", "size"),
    median_difference=("difference", "median"),
    positive_year_fraction=("difference", lambda x: (x > 0).mean()),
)
print(district_summary.round(3).to_string())

district_summary.to_csv(
    "research_outputs/mh_dengue_september_district_summary.csv"
)
print("\nSaved research_outputs/mh_dengue_september_district_summary.csv")
