import pandas as pd
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

raw = pd.read_csv("data/Final_data.csv")
raw = raw[
    raw["state_ut"].astype(str).str.strip().str.lower().eq("maharashtra")
    & raw["Disease"].astype(str).str.strip().str.casefold().eq(
        "Acute Diarrhoeal Disease".casefold()
    )
].copy()

for col in ["Cases", "Temp", "preci", "LAI", "year", "mon", "day"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

raw["date"] = pd.to_datetime(
    dict(year=raw["year"], month=raw["mon"], day=raw["day"]),
    errors="coerce"
)

daily = pd.read_csv(OUT / "mh_add_district_date.csv")
daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

# Select high observations from the deduplicated dataset.
top = daily.nlargest(15, "Cases")[["district", "date", "Cases"]].copy()

# Inspect matching raw rows without discarding duplicate records.
details = raw.merge(
    top[["district", "date"]],
    on=["district", "date"],
    how="inner",
    suffixes=("_raw", "_selected")
)

print("=== TOP 15 DEDUPLICATED OBSERVATIONS ===")
print(top.sort_values("Cases", ascending=False).to_string(index=False))

print("\n=== MATCHING RAW RECORDS ===")
cols = [
    "district", "date", "Cases",
    "Temp", "preci", "LAI", "week_of_outbreak"
]
cols = [c for c in cols if c in details.columns]
print(details[cols].sort_values(
    ["date", "district", "Cases"], ascending=[True, True, False]
).to_string(index=False))

# Show the count distribution in the districts with the largest observations.
districts = top["district"].unique()
print("\n=== ALL OBSERVATIONS IN DISTRICTS WITH TOP COUNTS ===")
subset = daily[daily["district"].isin(districts)].sort_values(
    ["district", "date"]
)
print(subset.groupby("district").agg(
    observations=("Cases", "size"),
    median_cases=("Cases", "median"),
    maximum_cases=("Cases", "max"),
    first_date=("date", "min"),
    last_date=("date", "max"),
).round(2).to_string())

details.to_csv(OUT / "mh_add_extreme_raw_records.csv", index=False)
top.to_csv(OUT / "mh_add_top15_observations.csv", index=False)
print("\nSaved extreme-observation audit files in research_outputs/")
