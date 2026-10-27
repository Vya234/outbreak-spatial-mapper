
import pandas as pd
from pathlib import Path

SOURCE = Path("data_corrected/Final_data_standardized.csv")
OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(SOURCE)
df.columns = df.columns.str.strip()

df = df[
    df["state_ut"].astype(str).str.strip().str.casefold().eq("maharashtra")
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
df["district"] = df["district"].astype(str).str.strip()

print("=== STANDARDIZED MAHARASHTRA ADD DATA ===")
print("Raw rows:", len(df))
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

# Preserve the existing analysis rule: median per district-date.
daily = (
    df.groupby(["district", "date"], as_index=False)
    .agg(
        Cases=("Cases", "median"),
        Temp=("Temp", "median"),
        preci=("preci", "median"),
        LAI=("LAI", "median"),
    )
)

print("\n=== STANDARDIZATION CHECK ===")
print("Solapur ADD observations:",
      int(daily["district"].str.casefold().eq("solapur").sum()))
print("District count:", daily["district"].nunique())
print("Analysis rows:", len(daily))

# Write to new files; do not overwrite previous results.
daily_path = OUT / "mh_add_district_date_standardized.csv"
daily.to_csv(daily_path, index=False)

yearly = daily.groupby(daily["date"].dt.year).agg(
    observations=("Cases", "count"),
    districts=("district", "nunique"),
    median_cases=("Cases", "median"),
    total_recorded_cases=("Cases", "sum"),
)
yearly_path = OUT / "mh_add_yearly_summary_standardized.csv"
yearly.to_csv(yearly_path)

print("\nSaved:", daily_path)
print("Saved:", yearly_path)
print("Original source and previous analysis outputs were not modified.")
