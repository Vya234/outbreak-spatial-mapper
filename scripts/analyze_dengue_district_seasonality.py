from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("research_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Load the exact-label Maharashtra Dengue subset
df = pd.read_csv("data/Final_data.csv")

for col in ["state_ut", "Disease", "district"]:
    df[col] = df[col].astype("string").str.strip()

d = df[
    (df["state_ut"] == "Maharashtra")
    & (df["Disease"] == "Dengue")
].copy()

for col in ["day", "mon", "year", "Cases"]:
    d[col] = pd.to_numeric(d[col], errors="coerce")

d["report_date"] = pd.to_datetime(
    {"year": d["year"], "month": d["mon"], "day": d["day"]},
    errors="coerce",
)

d = d.dropna(subset=["district", "report_date", "Cases"]).copy()
d["month"] = d["report_date"].dt.month

# One value per district-date for this sensitivity analysis.
# Median is used because conflicting repeated case counts exist.
daily = (
    d.groupby(["district", "report_date"], as_index=False)
    .agg(cases=("Cases", "median"))
)
daily["month"] = daily["report_date"].dt.month

# District-month summaries across the available years
district_month = (
    daily.groupby(["district", "month"])
    .agg(
        distinct_dates=("report_date", "nunique"),
        median_cases=("cases", "median"),
        mean_cases=("cases", "mean"),
    )
    .reset_index()
)

district_month.to_csv(
    OUT / "mh_dengue_district_month_summary.csv", index=False
)

# Wide table of district medians by calendar month
wide = district_month.pivot(
    index="district", columns="month", values="median_cases"
)
counts = district_month.pivot(
    index="district", columns="month", values="distinct_dates"
)

# Per-district comparison: September median vs. median of all other
# observed months. These are descriptive comparisons, not paired tests.
results = []

for district in wide.index:
    row = wide.loc[district]
    september = row.get(9, np.nan)
    other_months = row.drop(labels=[9], errors="ignore").dropna()

    if pd.notna(september) and len(other_months) > 0:
        other_median = other_months.median()
        results.append({
            "district": district,
            "september_median": september,
            "median_of_other_observed_month_medians": other_median,
            "difference_september_minus_other_months": (
                september - other_median
            ),
            "september_dates": int(counts.loc[district, 9]),
            "other_months_available": len(other_months),
            "september_higher": september > other_median,
            "september_equal": september == other_median,
            "september_lower": september < other_median,
        })

comparison = pd.DataFrame(results)
comparison.to_csv(
    OUT / "mh_dengue_september_district_comparison.csv", index=False
)

print("=== DISTRICT-MONTH COVERAGE ===")
print("Districts in dataset:", daily["district"].nunique())
print("Unique district-date observations:", len(daily))
print("Districts with September observations:",
      int(wide[9].notna().sum()) if 9 in wide.columns else 0)

print("\n=== SEPTEMBER VS OTHER OBSERVED MONTHS ===")
if comparison.empty:
    print("No districts have both September and another observed month.")
else:
    print(comparison.sort_values(
        "difference_september_minus_other_months",
        ascending=False
    ).round(2).to_string(index=False))

    print("\nDistrict comparison counts:")
    print("September higher:",
          int(comparison["september_higher"].sum()))
    print("Equal:",
          int(comparison["september_equal"].sum()))
    print("September lower:",
          int(comparison["september_lower"].sum()))

# Plot monthly district medians. Each month includes only districts with data.
monthly_values = [
    district_month.loc[
        district_month["month"] == month, "median_cases"
    ].dropna().to_numpy()
    for month in range(1, 13)
]

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(
    monthly_values,
    tick_labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    showfliers=False,
)
ax.set_title("District-Level Monthly Median Cases: Maharashtra Dengue")
ax.set_xlabel("Calendar month")
ax.set_ylabel("District median cases per observation")
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.text(
    0.5, 0.01,
    "Each box uses only districts with observations in that month; "
    "district coverage varies.",
    ha="center", fontsize=9
)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(
    OUT / "mh_dengue_district_month_boxplot.png",
    dpi=200, bbox_inches="tight"
)
plt.close(fig)

print("\nSaved outputs:")
print(OUT / "mh_dengue_district_month_summary.csv")
print(OUT / "mh_dengue_september_district_comparison.csv")
print(OUT / "mh_dengue_district_month_boxplot.png")
