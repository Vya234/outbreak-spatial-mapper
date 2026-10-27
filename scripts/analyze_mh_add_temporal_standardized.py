import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

INPUT = OUT / "mh_add_district_date_standardized.csv"

YEARLY_OUTPUT = OUT / "mh_add_temporal_yearly_standardized.csv"
MONTHLY_OUTPUT = OUT / "mh_add_temporal_monthly_standardized.csv"
FIGURE_OUTPUT = OUT / "mh_add_temporal_patterns_standardized.png"

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")

df = df.dropna(subset=["date", "district", "Cases"]).copy()
df["district"] = df["district"].astype(str).str.strip()
df = df[df["district"] != ""].copy()

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

yearly.to_csv(YEARLY_OUTPUT, index=False)

print("=== YEARLY SUMMARY ===")
print(yearly.round(2).to_string(index=False))

# Month-of-year summary across all years.
monthly = (
    df.groupby("month")
    .agg(
        observations=("Cases", "size"),
        districts=("district", "nunique"),
        median_cases=("Cases", "median"),
        mean_cases=("Cases", "mean"),
    )
    .reindex(range(1, 13))
    .rename_axis("month")
    .reset_index()
)

monthly.to_csv(MONTHLY_OUTPUT, index=False)

print("\n=== MONTH-OF-YEAR SUMMARY ===")
print(monthly.round(2).to_string(index=False))

# Coverage diagnostic: how many distinct districts were observed each year?
coverage = (
    df.groupby("year")
    .agg(
        observed_districts=("district", "nunique"),
        observed_records=("district", "size"),
    )
    .reset_index()
)

fig, axes = plt.subplots(3, 1, figsize=(11, 12))

axes[0].plot(
    yearly["year"], yearly["observations"], marker="o"
)
axes[0].set_title("Maharashtra ADD: Observed Records per Year")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("District-date records")
axes[0].grid(alpha=0.3)

axes[1].plot(
    monthly["month"], monthly["median_cases"], marker="o"
)
axes[1].set_xticks(range(1, 13))
axes[1].set_title("Maharashtra ADD: Median Recorded Cases by Calendar Month")
axes[1].set_xlabel("Calendar month")
axes[1].set_ylabel("Median recorded cases")
axes[1].grid(alpha=0.3)

axes[2].plot(
    coverage["year"], coverage["observed_districts"], marker="o"
)
axes[2].set_title("Maharashtra ADD: District Reporting Coverage")
axes[2].set_xlabel("Year")
axes[2].set_ylabel("Distinct districts observed")
axes[2].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIGURE_OUTPUT, dpi=160)
plt.close(fig)

print("\nSaved:", YEARLY_OUTPUT)
print("Saved:", MONTHLY_OUTPUT)
print("Saved:", FIGURE_OUTPUT)
print(
    "\nCaution: these are recorded observations. Changes in reporting "
    "coverage, duplicate/reporting semantics, and uneven sampling can "
    "affect apparent trends. The plots do not establish true incidence "
    "trends or prove seasonality."
)
