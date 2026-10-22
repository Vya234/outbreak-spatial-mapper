import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/Final_data.csv")

df = df[
    df["state_ut"].astype(str).str.strip().str.lower().eq("maharashtra")
    & df["Disease"].astype(str).str.strip().eq("Dengue")
].copy()

climate_cols = ["Temp", "preci", "LAI"]

for col in climate_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("=== DATASET ===")
print("Rows:", len(df))
print("Districts:", df["district"].nunique())
print("Date range fields:")
print(df[["year", "mon", "day"]].describe().round(2).to_string())

print("\n=== CLIMATE VARIABLE AUDIT ===")

summary = []

for col in climate_cols:
    s = df[col]
    valid = s.dropna()

    if len(valid) == 0:
        print(f"\n{col}: no numeric observations")
        continue

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1

    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr

    # These are statistical flags, not proof of erroneous measurements.
    outliers = valid[(valid < low_fence) | (valid > high_fence)]

    summary.append({
        "variable": col,
        "rows": len(s),
        "missing": int(s.isna().sum()),
        "missing_pct": 100 * s.isna().mean(),
        "unique_values": int(valid.nunique()),
        "mean": valid.mean(),
        "std": valid.std(),
        "min": valid.min(),
        "q1": q1,
        "median": valid.median(),
        "q3": q3,
        "max": valid.max(),
        "iqr_outlier_count": len(outliers),
    })

    print(f"\n--- {col} ---")
    print("Missing:", int(s.isna().sum()), "/", len(s),
          f"({100 * s.isna().mean():.1f}%)")
    print("Unique numeric values:", valid.nunique())
    print("Mean:", round(valid.mean(), 3))
    print("Standard deviation:", round(valid.std(), 3))
    print("Min / median / max:",
          round(valid.min(), 3),
          round(valid.median(), 3),
          round(valid.max(), 3))
    print("IQR outlier flags:", len(outliers))

    print("\nLowest five values:")
    print(valid.nsmallest(5).to_string(index=False))
    print("\nHighest five values:")
    print(valid.nlargest(5).to_string(index=False))

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT / "mh_dengue_climate_audit.csv", index=False)

print("\n=== CLIMATE VARIATION BY DISTRICT ===")
for col in climate_cols:
    district_stats = df.groupby("district")[col].agg(
        observations="count",
        unique_values="nunique",
        minimum="min",
        maximum="max",
        std="std",
    )
    print(f"\n--- {col} ---")
    print(district_stats.describe().round(3).to_string())

print("\n=== CLIMATE VARIABLE CORRELATIONS ===")
print(df[climate_cols].corr(method="spearman").round(3).to_string())

print("\nSaved:", OUT / "mh_dengue_climate_audit.csv")
print("Note: IQR outlier flags require investigation; they are not automatic errors.")
print("Climate values may be repeated across records and dates.")
