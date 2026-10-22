import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs"

path = OUT / "dengue_district_climate_correlations.csv"
df = pd.read_csv(path)

# Benjamini-Hochberg false discovery rate correction.
p = df["p_value"]
rank = p.rank(method="first")
m = len(p)

df["p_adjusted_BH"] = (p * m / rank).clip(upper=1)

# Enforce monotonic adjusted p-values in sorted p-value order.
order = p.sort_values().index
adjusted_sorted = (
    p.loc[order] * m /
    pd.Series(range(1, m + 1), index=order)
).iloc[::-1].cummin().iloc[::-1].clip(upper=1)

df["p_adjusted_BH"] = adjusted_sorted.reindex(df.index)
df["significant_BH_0.05"] = df["p_adjusted_BH"] < 0.05

df = df.sort_values("p_adjusted_BH")
df.to_csv(OUT / "dengue_district_climate_correlations_corrected.csv",
          index=False)

print("Total tests:", len(df))
print("Significant after BH correction:",
      int(df["significant_BH_0.05"].sum()))

print("\nSmallest adjusted p-values:")
print(
    df[["district", "Feature", "N", "Spearman_rho",
        "p_value", "p_adjusted_BH", "significant_BH_0.05"]]
    .head(15).round(4).to_string(index=False)
)

print("\nSaved corrected results.")
