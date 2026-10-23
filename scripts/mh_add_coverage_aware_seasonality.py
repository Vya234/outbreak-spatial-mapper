import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("research_outputs")
df = pd.read_csv(OUT / "mh_add_district_date.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["date", "district", "Cases"]).copy()

# Normalize a known district-name variant.
df["district"] = df["district"].astype(str).str.strip()
df["district"] = df["district"].replace({
    "Sholapur": "Solapur",
    "sholapur": "Solapur",
})

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Each district's recorded-case median for each calendar month.
district_month = (
    df.groupby(["district", "month"])
    .agg(
        median_cases=("Cases", "median"),
        observations=("Cases", "size"),
        years_observed=("year", "nunique"),
    )
    .reset_index()
)

# Require a reasonable spread of months and years before including a district.
district_coverage = df.groupby("district").agg(
    months_observed=("month", "nunique"),
    years_observed=("year", "nunique"),
)

eligible = district_coverage[
    (district_coverage["months_observed"] >= 6)
    & (district_coverage["years_observed"] >= 3)
].index

balanced = district_month[
    district_month["district"].isin(eligible)
].copy()

summary = (
    balanced.groupby("month")
    .agg(
        contributing_districts=("district", "nunique"),
        median_of_district_medians=("median_cases", "median"),
        lower_quartile=("median_cases", lambda x: x.quantile(0.25)),
        upper_quartile=("median_cases", lambda x: x.quantile(0.75)),
    )
    .reindex(range(1, 13))
    .rename_axis("month")
    .reset_index()
)

summary.to_csv(
    OUT / "mh_add_coverage_aware_seasonality.csv", index=False
)

print("Eligible districts:", len(eligible))
print("Districts in source:", df["district"].nunique())
print("\n=== COVERAGE-AWARE MONTHLY SUMMARY ===")
print(summary.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
x = summary["month"]
y = summary["median_of_district_medians"]

ax.plot(x, y, marker="o")
ax.fill_between(
    x,
    summary["lower_quartile"],
    summary["upper_quartile"],
    alpha=0.2,
)
ax.set_xticks(range(1, 13))
ax.set_xlabel("Calendar month")
ax.set_ylabel("Median of district-level median recorded cases")
ax.set_title("Maharashtra ADD: Coverage-Aware Descriptive Seasonality")
ax.grid(alpha=0.3)
fig.tight_layout()

path = OUT / "mh_add_coverage_aware_seasonality.png"
fig.savefig(path, dpi=160)
plt.close(fig)

print("\nSaved CSV:", OUT / "mh_add_coverage_aware_seasonality.csv")
print("Saved plot:", path)
print(
    "\nInterpretation: descriptive comparison only. "
    "This does not correct for unobserved reports or prove seasonality."
)
