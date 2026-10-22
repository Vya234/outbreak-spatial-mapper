import pandas as pd
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("data/Final_data.csv")

df = df[
    df["state_ut"].astype(str).str.strip().str.lower().eq("west bengal")
    & df["Disease"].astype(str).str.strip().eq("Acute Diarrhoeal Disease")
].copy()

for col in ["Cases", "Temp", "preci", "LAI", "year", "mon", "day"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["date"] = pd.to_datetime(
    dict(year=df["year"], month=df["mon"], day=df["day"]),
    errors="coerce"
)
df = df.dropna(subset=["date", "district"]).copy()

print("=== OVERALL COVERAGE ===")
print("Rows:", len(df))
print("Districts:", df["district"].nunique())
print("Distinct dates:", df["date"].nunique())
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
print("Missing Cases:", df["Cases"].isna().sum())
print("Missing Temp:", df["Temp"].isna().sum())
print("Missing precipitation:", df["preci"].isna().sum())
print("Missing LAI:", df["LAI"].isna().sum())

print("\n=== ROWS PER YEAR ===")
print(df.groupby(df["date"].dt.year).size().to_string())

# Check repeated district-date records before aggregating.
dup = df[df.duplicated(["district", "date"], keep=False)].copy()
print("\n=== DUPLICATE DISTRICT-DATE AUDIT ===")
print("Rows belonging to duplicate groups:", len(dup))
print("Duplicate district-date groups:",
      dup.groupby(["district", "date"]).ngroups)

if not dup.empty:
    duplicate_summary = (
        dup.groupby(["district", "date"])
        .agg(
            rows=("Cases", "size"),
            distinct_case_values=("Cases", "nunique")
        )
        .reset_index()
    )
    print("Duplicate groups with conflicting Cases:",
          (duplicate_summary["distinct_case_values"] > 1).sum())
    duplicate_summary.to_csv(
        OUT / "wb_add_duplicate_district_dates.csv", index=False
    )

# Use median only for the coverage audit, not as a claim about true incidence.
daily = (
    df.groupby(["district", "date"], as_index=False)
    .agg(
        Cases=("Cases", "median"),
        Temp=("Temp", "median"),
        preci=("preci", "median"),
        LAI=("LAI", "median")
    )
)

print("\n=== DISTRICT COVERAGE ===")
district_rows = []
for district, g in daily.groupby("district"):
    dates = g["date"].sort_values()
    gaps = dates.diff().dt.days.dropna()
    span_days = (dates.max() - dates.min()).days + 1

    district_rows.append({
        "district": district,
        "observations": len(g),
        "first_date": dates.min(),
        "last_date": dates.max(),
        "span_days": span_days,
        "median_gap_days": gaps.median() if len(gaps) else None,
        "max_gap_days": gaps.max() if len(gaps) else None,
        "observed_week_count": g["date"].dt.to_period("W-SUN").nunique(),
    })

district_summary = pd.DataFrame(district_rows)
district_summary.to_csv(OUT / "wb_add_district_coverage.csv", index=False)

for col in ["observations", "median_gap_days", "max_gap_days"]:
    print(f"{col} — median:", district_summary[col].median(),
          "| min:", district_summary[col].min(),
          "| max:", district_summary[col].max())

# Count district-weeks with observations; this is not a complete panel.
daily["week"] = daily["date"].dt.to_period("W-SUN")
weekly = daily.groupby(["district", "week"]).size().reset_index(name="dates")
print("\n=== WEEKLY COVERAGE ===")
print("Observed district-week combinations:", len(weekly))
print("Distinct calendar weeks:", weekly["week"].nunique())
print("District-weeks with multiple observed dates:",
      (weekly["dates"] > 1).sum())

print("\n=== CASE COUNT SUMMARY ===")
print(daily["Cases"].describe(percentiles=[0.25, 0.5, 0.75]).round(2).to_string())
print("Zero case observations:", (daily["Cases"] == 0).sum())
print("Negative case observations:", (daily["Cases"] < 0).sum())

print("\nSaved:", OUT / "wb_add_district_coverage.csv")
print("Duplicate audit, if applicable:",
      OUT / "wb_add_duplicate_district_dates.csv")
