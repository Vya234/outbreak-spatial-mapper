import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/Final_data.csv")

df = df[
    df["state_ut"].astype(str).str.strip().str.lower().eq("maharashtra")
    & df["Disease"].astype(str).str.strip().eq("Dengue")
].copy()

df["report_date"] = pd.to_datetime(
    {
        "year": pd.to_numeric(df["year"], errors="coerce"),
        "month": pd.to_numeric(df["mon"], errors="coerce"),
        "day": pd.to_numeric(df["day"], errors="coerce"),
    },
    errors="coerce",
)

df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["report_date", "Cases", "district"])
df = df[df["Cases"] >= 0].copy()
df["district"] = df["district"].astype(str).str.strip()

# Same deduplication rule used in previous experiments.
daily = (
    df.groupby(["district", "report_date"], as_index=False)
    .agg(Cases=("Cases", "median"))
)

daily["year"] = daily["report_date"].dt.year
daily["month"] = daily["report_date"].dt.month

# A. Monthly pooled descriptive statistics.
monthly = (
    daily.groupby("month")["Cases"]
    .agg(
        observations="size",
        median_cases="median",
        mean_cases="mean",
    )
    .reindex(range(1, 13))
)

monthly.index.name = "month"
print("\n=== MONTHLY RECORDED CASE COUNTS ===")
print(monthly.round(2).to_string())

monthly.to_csv(OUT / "mh_dengue_monthly_summary.csv")

# B. District-year comparisons.
# Compare each district-year's September median with its median
# across other months that have at least one recorded observation.
rows = []

for (district, year), g in daily.groupby(["district", "year"]):
    september = g.loc[g["month"] == 9, "Cases"]
    other = g.loc[g["month"] != 9, "Cases"]

    if len(september) == 0 or len(other) == 0:
        continue

    sep_median = september.median()
    other_median = other.median()

    rows.append({
        "district": district,
        "year": year,
        "september_observations": len(september),
        "other_month_observations": len(other),
        "september_median": sep_median,
        "other_month_median": other_median,
        "difference": sep_median - other_median,
        "september_higher": sep_median > other_median,
    })

comparisons = pd.DataFrame(rows)

print("\n=== SEPTEMBER VS OTHER OBSERVED MONTHS WITHIN DISTRICT-YEARS ===")
if comparisons.empty:
    print("No eligible district-year groups.")
else:
    print("Eligible district-year groups:", len(comparisons))
    print("September higher:", int(comparisons["september_higher"].sum()))
    print("September equal:", int((comparisons["difference"] == 0).sum()))
    print("September lower:", int((comparisons["difference"] < 0).sum()))
    print("Median difference:", round(comparisons["difference"].median(), 2))
    print("\nResults by year:")
    print(
        comparisons.groupby("year").agg(
            district_year_groups=("difference", "size"),
            median_difference=("difference", "median"),
            positive_groups=("september_higher", "sum"),
        ).round(2).to_string()
    )

comparisons.to_csv(
    OUT / "mh_dengue_september_district_year_comparison.csv",
    index=False,
)

# C. Monthly median plot, accompanied by sample counts.
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    monthly.index,
    monthly["median_cases"],
    marker="o",
    label="Monthly median",
)
ax.set_xticks(range(1, 13))
ax.set_xlabel("Calendar month")
ax.set_ylabel("Recorded cases per observation")
ax.set_title("Maharashtra Dengue: Monthly Recorded Case Counts")
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.bar(
    monthly.index,
    monthly["observations"],
    alpha=0.2,
    label="Number of observations",
)
ax2.set_ylabel("Number of observations")

fig.tight_layout()
fig.savefig(
    OUT / "mh_dengue_monthly_seasonality.png",
    dpi=180,
)
plt.close(fig)

# D. Year-by-month heatmap of medians.
heat = daily.pivot_table(
    index="year",
    columns="month",
    values="Cases",
    aggfunc="median",
).reindex(columns=range(1, 13))

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(heat.values, aspect="auto", interpolation="nearest")
ax.set_xticks(range(12))
ax.set_xticklabels(range(1, 13))
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index)
ax.set_xlabel("Calendar month")
ax.set_ylabel("Year")
ax.set_title("Monthly Median Recorded Dengue Cases by Year")
fig.colorbar(im, ax=ax, label="Median recorded cases")
fig.tight_layout()
fig.savefig(
    OUT / "mh_dengue_year_month_heatmap.png",
    dpi=180,
)
plt.close(fig)

# Save heatmap values as well, so missing cells remain inspectable.
heat.to_csv(OUT / "mh_dengue_year_month_median.csv")

print("\nSaved:")
print(OUT / "mh_dengue_monthly_summary.csv")
print(OUT / "mh_dengue_september_district_year_comparison.csv")
print(OUT / "mh_dengue_monthly_seasonality.png")
print(OUT / "mh_dengue_year_month_heatmap.png")
print(OUT / "mh_dengue_year_month_median.csv")

print("\nCaution: these are descriptive comparisons of recorded cases.")
print("Uneven reporting means the patterns are not proof of incidence seasonality.")
