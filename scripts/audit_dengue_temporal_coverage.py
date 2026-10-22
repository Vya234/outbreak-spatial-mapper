import pandas as pd
import numpy as np
from pathlib import Path

INPUT = "data/Final_data.csv"
OUT = Path("research_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load and filter the exact disease label.
df = pd.read_csv(INPUT)

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
df["district"] = df["district"].astype("string").str.strip()

df = df.dropna(subset=["report_date", "district", "Cases"])
df = df[df["Cases"] >= 0].copy()

print("=" * 65)
print("MAHARASHTRA DENGUE: TEMPORAL COVERAGE AUDIT")
print("=" * 65)
print("Valid records:", len(df))
print("Districts:", df["district"].nunique())
print("Date range:", df["report_date"].min().date(), "to",
      df["report_date"].max().date())

# 2. Audit duplicate district-date records before aggregation.
duplicate_groups = (
    df.groupby(["district", "report_date"])
    .size()
    .rename("rows")
    .reset_index()
)
duplicate_groups = duplicate_groups[duplicate_groups["rows"] > 1]

print("\n=== DUPLICATE REPORTING DATES ===")
print("Repeated district-date groups:", len(duplicate_groups))
print("Rows belonging to those groups:",
      int(duplicate_groups["rows"].sum()))

if len(duplicate_groups):
    print(duplicate_groups.sort_values(
        "rows", ascending=False
    ).head(15).to_string(index=False))

# Collapse duplicates to one district-date observation using the
# same median rule as the earlier modeling experiment.
daily = (
    df.groupby(["district", "report_date"], as_index=False)
    .agg(
        cases=("Cases", "median"),
        source_rows=("Cases", "size"),
    )
)

# 3. Assign each observed date to its ISO calendar week.
iso = daily["report_date"].dt.isocalendar()
daily["iso_year"] = iso.year.astype(int)
daily["iso_week"] = iso.week.astype(int)
daily["week_start"] = (
    daily["report_date"]
    - pd.to_timedelta(daily["report_date"].dt.weekday, unit="D")
)

# This is a coverage aggregation only. It does NOT establish that
# summed records represent complete weekly incidence.
weekly = (
    daily.groupby(["district", "week_start"], as_index=False)
    .agg(
        observed_dates=("report_date", "nunique"),
        daily_observations=("cases", "size"),
        summed_recorded_cases=("cases", "sum"),
    )
)

print("\n=== OBSERVED WEEKLY COVERAGE ===")
print("District-week combinations with >=1 observed date:", len(weekly))
print("Distinct observed calendar weeks:", weekly["week_start"].nunique())
print("District-weeks with multiple observed dates:",
      int((weekly["observed_dates"] > 1).sum()))

print("\nObserved dates per district-week:")
print(weekly["observed_dates"].value_counts().sort_index().to_string())

# 4. Create a coverage grid per district to measure missing weeks.
# Missing weeks here mean no row was recorded in the source for that
# district-week; they are NOT imputed as zero cases.
coverage_rows = []

for district, g in weekly.groupby("district"):
    g = g.sort_values("week_start")
    observed_weeks = pd.DatetimeIndex(g["week_start"].unique()).sort_values()

    full_weeks = pd.date_range(
        observed_weeks.min(),
        observed_weeks.max(),
        freq="7D",
    )

    observed_set = set(observed_weeks)
    flags = np.array([w in observed_set for w in full_weeks], dtype=bool)

    n_observed = int(flags.sum())
    n_possible = len(flags)
    missing_weeks = n_possible - n_observed

    # Longest run of consecutive weeks with observations.
    longest_run = 0
    current_run = 0
    for flag in flags:
        if flag:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    # Number of observed weeks that have an immediately preceding
    # observed week, useful as a simple one-week-lag coverage measure.
    consecutive_pairs = int(np.sum(flags[1:] & flags[:-1]))

    coverage_rows.append({
        "district": district,
        "first_observed_week": observed_weeks.min().date(),
        "last_observed_week": observed_weeks.max().date(),
        "observed_weeks": n_observed,
        "possible_weeks_in_span": n_possible,
        "missing_weeks_in_span": missing_weeks,
        "coverage_fraction": n_observed / n_possible,
        "longest_consecutive_observed_weeks": longest_run,
        "consecutive_week_pairs": consecutive_pairs,
    })

coverage = pd.DataFrame(coverage_rows).sort_values(
    "observed_weeks", ascending=False
)

print("\n=== DISTRICT-LEVEL COVERAGE SUMMARY ===")
for col in [
    "observed_weeks",
    "possible_weeks_in_span",
    "coverage_fraction",
    "longest_consecutive_observed_weeks",
    "consecutive_week_pairs",
]:
    print(f"\n{col}:")
    print(coverage[col].describe().round(3).to_string())

print("\nDistricts with the most observed weeks:")
print(coverage.head(15).round(3).to_string(index=False))

print("\n=== DISTRICTS WITH AT LEAST 8 CONSECUTIVE OBSERVED WEEKS ===")
eligible = coverage[
    coverage["longest_consecutive_observed_weeks"] >= 8
]
print("Count:", len(eligible))
if len(eligible):
    print(eligible[
        ["district", "observed_weeks",
         "coverage_fraction", "longest_consecutive_observed_weeks"]
    ].round(3).to_string(index=False))

# 5. Gap between consecutive observed dates within each district.
gap_rows = []

for district, g in daily.groupby("district"):
    dates = g["report_date"].sort_values().drop_duplicates()
    gaps = dates.diff().dt.days.dropna()

    if len(gaps):
        gap_rows.append({
            "district": district,
            "number_of_date_gaps": len(gaps),
            "median_gap_days": gaps.median(),
            "mean_gap_days": gaps.mean(),
            "maximum_gap_days": gaps.max(),
            "gaps_le_14_days": int((gaps <= 14).sum()),
            "gaps_le_7_days": int((gaps <= 7).sum()),
        })

gaps = pd.DataFrame(gap_rows)

print("\n=== BETWEEN-REPORT DATE GAPS ===")
if len(gaps):
    print(gaps[
        ["median_gap_days", "mean_gap_days", "maximum_gap_days",
         "gaps_le_14_days", "gaps_le_7_days"]
    ].describe().round(2).to_string())

    print("\nDistricts with shortest median gaps:")
    print(gaps.sort_values("median_gap_days").head(10)
          .round(2).to_string(index=False))

# 6. Save auditable outputs.
daily.to_csv(OUT / "mh_dengue_district_daily_deduplicated.csv", index=False)
weekly.to_csv(OUT / "mh_dengue_observed_weekly_coverage.csv", index=False)
coverage.to_csv(OUT / "mh_dengue_district_week_coverage.csv", index=False)
gaps.to_csv(OUT / "mh_dengue_reporting_gaps.csv", index=False)

print("\n=== SAVED FILES ===")
for name in [
    "mh_dengue_district_daily_deduplicated.csv",
    "mh_dengue_observed_weekly_coverage.csv",
    "mh_dengue_district_week_coverage.csv",
    "mh_dengue_reporting_gaps.csv",
]:
    print(OUT / name)

print("\nIMPORTANT:")
print("Unobserved district-weeks are missing reports, not confirmed zero cases.")
print("Weekly sums are coverage diagnostics, not validated weekly incidence.")
