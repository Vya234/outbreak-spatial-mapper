from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

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

# One observation per district-date, using median for sensitivity analysis.
daily = (
    d.groupby(["district", "report_date"], as_index=False)
    .agg(cases=("Cases", "median"))
)
daily["month"] = daily["report_date"].dt.month
daily["year"] = daily["report_date"].dt.year

# Each district's monthly median and number of distinct reporting dates.
monthly = (
    daily.groupby(["district", "month"])
    .agg(
        median_cases=("cases", "median"),
        distinct_dates=("report_date", "nunique"),
    )
    .reset_index()
)

wide = monthly.pivot(
    index="district", columns="month", values="median_cases"
)
counts = monthly.pivot(
    index="district", columns="month", values="distinct_dates"
)

# Compare September to the median of the other observed monthly medians.
results = []

for district in wide.index:
    if 9 not in wide.columns or pd.isna(wide.loc[district, 9]):
        continue

    september_dates = int(counts.loc[district, 9])
    september_median = float(wide.loc[district, 9])

    other = wide.loc[district].drop(labels=[9], errors="ignore").dropna()
    if other.empty:
        continue

    results.append({
        "district": district,
        "september_dates": september_dates,
        "september_median": september_median,
        "other_months_median": float(other.median()),
        "difference": september_median - float(other.median()),
        "other_months_available": len(other),
    })

comp = pd.DataFrame(results)
comp.to_csv(OUT / "mh_dengue_september_validation_all.csv", index=False)

# Robustness subset: at least two distinct September reporting dates.
robust = comp[comp["september_dates"] >= 2].copy()
robust.to_csv(
    OUT / "mh_dengue_september_validation_min2.csv", index=False
)

print("=== ALL COMPARABLE DISTRICTS ===")
print("Districts:", len(comp))
if len(comp):
    print("Positive differences:", int((comp["difference"] > 0).sum()))
    print("Equal differences:", int((comp["difference"] == 0).sum()))
    print("Negative differences:", int((comp["difference"] < 0).sum()))
    print("Median difference:", round(comp["difference"].median(), 3))

print("\n=== DISTRICTS WITH AT LEAST 2 SEPTEMBER DATES ===")
print("Districts:", len(robust))
if len(robust):
    x = robust["difference"].to_numpy(dtype=float)
    print("Positive differences:", int((x > 0).sum()))
    print("Equal differences:", int((x == 0).sum()))
    print("Negative differences:", int((x < 0).sum()))
    print("Median difference:", round(float(np.median(x)), 3))

    rng = np.random.default_rng(42)
    B = 20000
    boot = np.array([
        np.median(rng.choice(x, size=len(x), replace=True))
        for _ in range(B)
    ])
    low, high = np.quantile(boot, [0.025, 0.975])
    print("Bootstrap 95% interval:",
          (round(float(low), 3), round(float(high), 3)))

    print("\nDistrict-level results:")
    print(robust.sort_values("difference", ascending=False)
          .round(2).to_string(index=False))
else:
    print("Too few districts to perform this comparison.")

# A second sensitivity analysis: compare months within each district-year.
# This reduces cross-year confounding but requires observed September and
# at least one other month in the same district-year.
district_year = (
    daily.groupby(["district", "year", "month"])["cases"]
    .median()
    .reset_index(name="median_cases")
)

within_results = []
for (district, year), group in district_year.groupby(["district", "year"]):
    sept = group.loc[group["month"] == 9, "median_cases"]
    other = group.loc[group["month"] != 9, "median_cases"]

    if len(sept) and len(other):
        within_results.append({
            "district": district,
            "year": year,
            "september_median": float(sept.iloc[0]),
            "other_observed_months_median": float(other.median()),
            "difference": float(sept.iloc[0] - other.median()),
        })

within = pd.DataFrame(within_results)
within.to_csv(
    OUT / "mh_dengue_september_within_district_year.csv", index=False
)

print("\n=== WITHIN-DISTRICT-YEAR COMPARISON ===")
print("District-year groups:", len(within))
if len(within):
    print("Positive:", int((within["difference"] > 0).sum()))
    print("Equal:", int((within["difference"] == 0).sum()))
    print("Negative:", int((within["difference"] < 0).sum()))
    print("Median difference:", round(within["difference"].median(), 3))

# Plot the district-level differences for the min-2 subset.
if len(robust):
    plot_data = robust.sort_values("difference")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_data["district"], plot_data["difference"])
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("September median minus other-month median (cases)")
    ax.set_ylabel("District")
    ax.set_title("September Robustness Check: Districts with ≥2 September Dates")
    fig.tight_layout()
    fig.savefig(
        OUT / "mh_dengue_september_robustness.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

print("\nOutputs saved in research_outputs/")
