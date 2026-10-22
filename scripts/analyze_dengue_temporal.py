from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/Final_data.csv")
OUT_DIR = Path("research_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

for col in ["state_ut", "Disease", "district"]:
    df[col] = df[col].astype("string").str.strip()

# Strictly isolate the 370 records labeled exactly "Dengue"
data = df[
    (df["state_ut"] == "Maharashtra")
    & (df["Disease"] == "Dengue")
].copy()

if data.empty:
    raise ValueError("No exact Maharashtra/Dengue records found.")

for col in ["Cases", "Deaths", "day", "mon", "year"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data["report_date"] = pd.to_datetime(
    {
        "year": data["year"],
        "month": data["mon"],
        "day": data["day"],
    },
    errors="coerce",
)

data = data.dropna(subset=["report_date"]).copy()
data["Cases"] = pd.to_numeric(data["Cases"], errors="coerce")
data["year"] = data["report_date"].dt.year
data["month"] = data["report_date"].dt.month

# Save filtered observations for reproducibility
data.to_csv(OUT_DIR / "mh_dengue_exact_observations.csv", index=False)

# Annual totals: sum recorded rows, not population-adjusted incidence
annual = (
    data.groupby("year")
    .agg(
        observation_rows=("Cases", "size"),
        distinct_reporting_dates=("report_date", "nunique"),
        total_recorded_cases=("Cases", "sum"),
        median_cases_per_observation=("Cases", "median"),
    )
    .reset_index()
)
annual.to_csv(OUT_DIR / "mh_dengue_annual_summary.csv", index=False)

# Monthly summary across all years
monthly = (
    data.groupby("month")
    .agg(
        observation_rows=("Cases", "size"),
        total_recorded_cases=("Cases", "sum"),
        median_cases_per_observation=("Cases", "median"),
    )
    .reindex(range(1, 13))
    .rename_axis("month")
    .reset_index()
)
monthly.to_csv(OUT_DIR / "mh_dengue_monthly_summary.csv", index=False)

# Plot annual recorded cases
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(annual["year"], annual["total_recorded_cases"])
ax.set_title("Maharashtra Dengue: Annual Recorded Cases")
ax.set_xlabel("Reporting year")
ax.set_ylabel("Sum of recorded case counts")
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "mh_dengue_annual_cases.png", dpi=200)
plt.close(fig)

# Plot monthly summaries across all observed years
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(monthly["month"], monthly["median_cases_per_observation"])
ax.set_xticks(range(1, 13))
ax.set_xlabel("Calendar month")
ax.set_ylabel("Median cases per observation")
ax.set_title("Maharashtra Dengue: Monthly Median Across Observations")
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "mh_dengue_monthly_median.png", dpi=200)
plt.close(fig)

# Reporting gaps within each district
gaps = []
for district, group in data.groupby("district"):
    dates = (
        group["report_date"]
        .drop_duplicates()
        .sort_values()
    )
    differences = dates.diff().dt.days.dropna()

    if len(differences):
        gaps.append({
            "district": district,
            "distinct_reporting_dates": len(dates),
            "median_gap_days": differences.median(),
            "maximum_gap_days": differences.max(),
        })

gap_summary = pd.DataFrame(gaps)
gap_summary.to_csv(
    OUT_DIR / "mh_dengue_reporting_gaps.csv", index=False
)

print("Exact Maharashtra Dengue records:", len(data))
print("Districts:", data["district"].nunique())
print(
    "Date range:",
    data["report_date"].min().date(),
    "to",
    data["report_date"].max().date(),
)

print("\nAnnual summary:")
print(annual.to_string(index=False))

print("\nMonthly summary:")
print(monthly.to_string(index=False))

print("\nReporting gap summary (days):")
if not gap_summary.empty:
    print(gap_summary[
        ["median_gap_days", "maximum_gap_days"]
    ].describe().round(2).to_string())

print("\nOutputs saved in research_outputs/")
