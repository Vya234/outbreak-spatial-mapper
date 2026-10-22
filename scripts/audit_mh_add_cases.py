import pandas as pd
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("data/Final_data.csv")
df = df[
    df["state_ut"].astype(str).str.strip().str.lower().eq("maharashtra")
    & df["Disease"].astype(str).str.strip().str.casefold().eq(
        "Acute Diarrhoeal Disease".casefold()
    )
].copy()

for col in ["Cases", "Temp", "preci", "LAI", "year", "mon", "day"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["date"] = pd.to_datetime(
    dict(year=df["year"], month=df["mon"], day=df["day"]),
    errors="coerce",
)
df = df.dropna(subset=["date", "district"]).copy()

print("=== RAW DATA ===")
print("Rows:", len(df))
print("Missing cases:", df["Cases"].isna().sum())
print("Zero cases:", (df["Cases"] == 0).sum())
print("Negative cases:", (df["Cases"] < 0).sum())

duplicates = (
    df.groupby(["district", "date"])
    .agg(rows=("Cases", "size"), distinct_cases=("Cases", "nunique"))
    .reset_index()
)
duplicates = duplicates[duplicates["rows"] > 1]

print("\n=== DUPLICATE AUDIT ===")
print("Duplicate district-date groups:", len(duplicates))
print("Groups with conflicting case values:",
      (duplicates["distinct_cases"] > 1).sum())
print("Rows in duplicate groups:", int(duplicates["rows"].sum()))

# Median is used to make a reproducible district-date analysis table.
daily = (
    df.groupby(["district", "date"], as_index=False)
    .agg(
        Cases=("Cases", "median"),
        Temp=("Temp", "median"),
        preci=("preci", "median"),
        LAI=("LAI", "median"),
    )
)

cases = daily["Cases"].dropna()

print("\n=== DEDUPLICATED CASE DISTRIBUTION ===")
print(cases.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).round(2))
print("Unique case values:", cases.nunique())
print("Top 10 case counts:")
print(cases.nlargest(10).to_string(index=False))

q1, q3 = cases.quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
print("IQR upper fence:", round(upper, 2))
print("Observations above upper fence:", int((cases > upper).sum()))

print("\n=== YEARLY COVERAGE AFTER DEDUPLICATION ===")
yearly = daily.groupby(daily["date"].dt.year).agg(
    observations=("Cases", "count"),
    districts=("district", "nunique"),
    median_cases=("Cases", "median"),
    total_recorded_cases=("Cases", "sum"),
)
print(yearly.round(2).to_string())

yearly.to_csv(OUT / "mh_add_yearly_summary.csv")
daily.to_csv(OUT / "mh_add_district_date.csv", index=False)

print("\nSaved:", OUT / "mh_add_yearly_summary.csv")
print("Saved:", OUT / "mh_add_district_date.csv")
