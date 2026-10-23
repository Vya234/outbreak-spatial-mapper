import pandas as pd
from pathlib import Path

OUT = Path("research_outputs")
df = pd.read_csv(OUT / "mh_add_district_date.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["date", "district", "Cases"]).copy()

df["district"] = df["district"].astype(str).str.strip()
df["district"] = df["district"].replace({
    "Sholapur": "Solapur",
    "sholapur": "Solapur",
})
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# One median per district-year-month, avoiding extra weight from
# districts that have more observations within a given month.
dym = (
    df.groupby(["district", "year", "month"])
    .agg(
        month_median=("Cases", "median"),
        observations=("Cases", "size"),
    )
    .reset_index()
)

# Only compare district-years with at least 4 distinct observed months.
coverage = dym.groupby(["district", "year"])["month"].nunique()
eligible = coverage[coverage >= 4].index
dym = dym.set_index(["district", "year"]).loc[
    lambda x: x.index.isin(eligible)
].reset_index()

baseline = (
    dym.groupby(["district", "year"])["month_median"]
    .median()
    .rename("district_year_baseline")
    .reset_index()
)

dym = dym.merge(baseline, on=["district", "year"], how="left")
dym["relative_to_baseline"] = (
    dym["month_median"] / dym["district_year_baseline"].where(
        dym["district_year_baseline"] > 0
    )
)

summary = (
    dym.groupby("month")
    .agg(
        district_years=("relative_to_baseline", "count"),
        median_relative_count=("relative_to_baseline", "median"),
        q25=("relative_to_baseline", lambda x: x.quantile(0.25)),
        q75=("relative_to_baseline", lambda x: x.quantile(0.75)),
    )
    .reindex(range(1, 13))
    .rename_axis("month")
    .reset_index()
)

summary.to_csv(
    OUT / "mh_add_seasonality_robustness.csv", index=False
)

print("Eligible district-years:", len(eligible))
print("\n=== MONTHLY RELATIVE-COUNT ROBUSTNESS ===")
print(summary.round(3).to_string(index=False))
print("\nSaved:", OUT / "mh_add_seasonality_robustness.csv")
print(
    "\nRatios are descriptive, not incidence-rate ratios. "
    "Sparse observations and reporting bias remain."
)
