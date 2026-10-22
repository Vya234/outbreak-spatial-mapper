import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(OUT / "mh_add_district_date.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["date", "Cases"]).copy()

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Yearly summary: recorded observations, not complete annual incidence.
yearly = (
    df.groupby("year")
    .agg(
        observations=("Cases", "size"),
        districts=("district", "nunique"),
        median_cases=("Cases", "median"),
        mean_cases=("Cases", "mean"),
        recorded_case_sum=("Cases", "sum"),
    )
    .reset_index()
)

yearly.to_csv(OUT / "mh_add_temporal_yearly.csv", index=False)
print("=== YEARLY SUMMARY ===")
print(yearly.round(2).to_string(index=False))

# Month-of-year summary across all years.
monthly = (
    df.groupby("month")
    .agg(
        observations=("Cases", "size"),
        median_cases=("Cases", "median"),
        mean_cases=("Cases", "mean"),
    )
    .reindex(range(1, 13))
    .rename_axis("month")
    .reset_index()
)

monthly.to_csv(OUT / "mh_add_temporal_monthly.csv", index=False)
print("\n=== MONTH-OF-YEAR SUMMARY ===")
print(monthly.round(2).to_string(index=False))

fig, axes = plt.subplots(2, 1, figsize=(11, 9))

axes[0].plot(
    yearly["year"], yearly["observations"], marker="o"
)
axes[0].set_title("Maharashtra ADD: Number of Observed Records per Year")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Observed district-date records")
axes[0].grid(alpha=0.3)

axes[1].plot(
    monthly["month"], monthly["median_cases"], marker="o"
)
axes[1].set_xticks(range(1, 13))
axes[1].set_title("Maharashtra ADD: Median Recorded Cases by Calendar Month")
axes[1].set_xlabel("Calendar month")
axes[1].set_ylabel("Median recorded cases")
axes[1].grid(alpha=0.3)

fig.tight_layout()
path = OUT / "mh_add_temporal_patterns.png"
fig.savefig(path, dpi=160)
plt.close(fig)

print("\nSaved:", OUT / "mh_add_temporal_yearly.csv")
print("Saved:", OUT / "mh_add_temporal_monthly.csv")
print("Saved:", path)
print(
    "\nCaution: reporting coverage varies over time; "
    "these plots do not establish a true incidence trend or seasonality."
)
