import pandas as pd
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("data/Final_data.csv")

for col in ["Cases", "Temp", "preci", "LAI", "year", "mon", "day"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["date"] = pd.to_datetime(
    dict(year=df["year"], month=df["mon"], day=df["day"]),
    errors="coerce",
)

candidates = [
    ("Maharashtra", "Acute Diarrhoeal Disease"),
    ("Madhya Pradesh", "Acute Diarrhoeal Disease"),
    ("Tamil Nadu", "Dengue"),
    ("Karnataka", "Dengue"),
    ("Karnataka", "Chikungunya"),
]

results = []

for state, disease in candidates:
    g = df[
        df["state_ut"].astype(str).str.strip().str.casefold().eq(state.casefold())
        & df["Disease"].astype(str).str.strip().str.casefold().eq(disease.casefold())
    ].dropna(subset=["date", "district"]).copy()

    # Count duplicates before aggregation.
    duplicate_groups = (
        g.groupby(["district", "date"]).size().gt(1).sum()
    )
    duplicate_rows = g.duplicated(
        ["district", "date"], keep=False
    ).sum()

    daily = (
        g.groupby(["district", "date"], as_index=False)
        .agg(
            Cases=("Cases", "median"),
            Temp=("Temp", "median"),
            preci=("preci", "median"),
            LAI=("LAI", "median"),
        )
    )

    district_obs = daily.groupby("district").size()
    gaps = []
    district_median_gaps = []

    for _, group in daily.groupby("district"):
        dates = group["date"].sort_values()
        gap = dates.diff().dt.days.dropna()
        if len(gap):
            gaps.extend(gap.tolist())
            district_median_gaps.append(gap.median())

    results.append({
        "state": state,
        "disease": disease,
        "raw_rows": len(g),
        "unique_district_dates": len(daily),
        "districts": daily["district"].nunique(),
        "distinct_dates": daily["date"].nunique(),
        "first_date": daily["date"].min(),
        "last_date": daily["date"].max(),
        "duplicate_groups": int(duplicate_groups),
        "rows_in_duplicate_groups": int(duplicate_rows),
        "districts_with_8plus_observations": int((district_obs >= 8).sum()),
        "median_observations_per_district": district_obs.median(),
        "median_of_district_median_gaps_days": (
            pd.Series(district_median_gaps).median()
            if district_median_gaps else None
        ),
        "temp_missing_pct_raw": round(g["Temp"].isna().mean() * 100, 1),
        "preci_missing_pct_raw": round(g["preci"].isna().mean() * 100, 1),
        "lai_missing_pct_raw": round(g["LAI"].isna().mean() * 100, 1),
    })

summary = pd.DataFrame(results).sort_values(
    ["districts_with_8plus_observations", "unique_district_dates"],
    ascending=False
)

summary.to_csv(OUT / "candidate_coverage_comparison.csv", index=False)
print(summary.to_string(index=False))
print("\nSaved:", OUT / "candidate_coverage_comparison.csv")
