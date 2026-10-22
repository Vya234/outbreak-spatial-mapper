from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

OUT = Path("research_outputs")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/Final_data.csv")

for col in ["state_ut", "Disease", "district"]:
    df[col] = df[col].astype("string").str.strip()

d = df[
    (df["state_ut"] == "Maharashtra")
    & (df["Disease"] == "Dengue")
].copy()

for col in ["Cases", "day", "mon", "year"]:
    d[col] = pd.to_numeric(d[col], errors="coerce")

d["report_date"] = pd.to_datetime(
    {"year": d["year"], "month": d["mon"], "day": d["day"]},
    errors="coerce",
)

d["Cases"] = pd.to_numeric(d["Cases"], errors="coerce")
d = d.dropna(subset=["district", "report_date"]).copy()
d["month"] = d["report_date"].dt.month
d["year"] = d["report_date"].dt.year

print("=== DATA OVERVIEW ===")
print("Rows:", len(d))
print("Districts:", d["district"].nunique())
print("Date range:", d["report_date"].min().date(), "to",
      d["report_date"].max().date())
print("Missing case values:", d["Cases"].isna().sum())
print("Negative case values:", (d["Cases"] < 0).sum())

# 1. Audit repeated district-date records.
keys = ["district", "report_date"]
duplicate_rows = d[d.duplicated(keys, keep=False)].sort_values(keys)

duplicate_rows.to_csv(
    OUT / "mh_dengue_duplicate_district_dates.csv", index=False
)

duplicate_summary = (
    d.groupby(keys, dropna=False)
    .agg(
        row_count=("Cases", "size"),
        distinct_case_values=("Cases", lambda s: s.nunique(dropna=True)),
        case_values=("Cases", lambda s: " | ".join(
            s.dropna().astype(str).tolist()
        )),
        missing_case_rows=("Cases", lambda s: s.isna().sum()),
        distinct_disease_labels=("Disease", "nunique"),
    )
    .reset_index()
)

duplicate_groups = duplicate_summary[
    duplicate_summary["row_count"] > 1
].copy()

duplicate_groups.to_csv(
    OUT / "mh_dengue_duplicate_group_summary.csv", index=False
)

print("\n=== DUPLICATE AUDIT ===")
print("Rows participating in repeated district-date groups:",
      len(duplicate_rows))
print("Repeated district-date groups:", len(duplicate_groups))
print("Groups with conflicting nonmissing case values:",
      (duplicate_groups["distinct_case_values"] > 1).sum())
print("Groups with missing case values:",
      (duplicate_groups["missing_case_rows"] > 0).sum())

# 2. Monthly summaries, including observation counts.
monthly = (
    d.groupby("month")
    .agg(
        observation_rows=("Cases", "size"),
        districts_observed=("district", "nunique"),
        median_cases=("Cases", "median"),
        mean_cases=("Cases", "mean"),
    )
    .reindex(range(1, 13))
    .rename_axis("month")
    .reset_index()
)

monthly.to_csv(OUT / "mh_dengue_monthly_seasonality_audit.csv",
               index=False)

print("\n=== POOLED MONTHLY SUMMARY ===")
print(monthly.round(2).to_string(index=False))

# 3. District-year-month cells.
# Aggregate repeated rows within the same district-date first, using the
# median only for this sensitivity analysis. This is NOT a claim that
# duplicate reports should be combined this way in the final dataset.
daily = (
    d.groupby(["district", "report_date"], as_index=False)
    .agg(cases=("Cases", "median"))
)
daily["year"] = daily["report_date"].dt.year
daily["month"] = daily["report_date"].dt.month

# Month-level median within each district-year. A cell is used only if
# at least one nonmissing case observation exists for that month.
cells = (
    daily.dropna(subset=["cases"])
    .groupby(["district", "year", "month"])["cases"]
    .median()
    .reset_index(name="median_cases")
)

cells.to_csv(OUT / "mh_dengue_district_year_month_cells.csv",
             index=False)

# 4. Friedman test: compare months within district-years that have
# observations in every one of the 12 months. This is intentionally
# conservative and may have very low power with sparse data.
wide = cells.pivot_table(
    index=["district", "year"],
    columns="month",
    values="median_cases",
    aggfunc="first",
)

complete = wide.dropna(subset=list(range(1, 13)))
print("\n=== WITHIN-DISTRICT-YEAR SEASONAL TEST ===")
print("District-year groups with all 12 months observed:", len(complete))

if len(complete) >= 2:
    arrays = [complete[m].to_numpy() for m in range(1, 13)]
    stat, pvalue = friedmanchisquare(*arrays)
    print(f"Friedman statistic: {stat:.4f}")
    print(f"p-value: {pvalue:.6g}")
    print("Interpret cautiously: complete cases may be highly selected.")
else:
    print("Insufficient complete district-year groups for a 12-month test.")
    print("No inferential conclusion about seasonality can be drawn from this test.")

# 5. Paired district-month comparison as an exploratory sensitivity check.
# First get each district's median for each calendar month across years.
dm = (
    cells.groupby(["district", "month"])["median_cases"]
    .median()
    .unstack("month")
)

print("\nDistricts with observations in each calendar month:")
if not dm.empty:
    print(dm.notna().sum().rename("districts_observed").to_string())

# Plot monthly medians and sample counts.
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly["month"], monthly["median_cases"], marker="o")
ax.set_xticks(range(1, 13))
ax.set_xlabel("Calendar month")
ax.set_ylabel("Median recorded cases per observation")
ax.set_title("Maharashtra Dengue: Pooled Monthly Median")
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "mh_dengue_monthly_median_audited.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(monthly["month"], monthly["observation_rows"])
ax.set_xticks(range(1, 13))
ax.set_xlabel("Calendar month")
ax.set_ylabel("Number of observations")
ax.set_title("Maharashtra Dengue: Monthly Observation Coverage")
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "mh_dengue_monthly_coverage_audited.png", dpi=200)
plt.close(fig)

print("\nOutputs saved in", OUT)
