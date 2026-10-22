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

group_cols = ["state_ut", "Disease"]
rows = []

for (state, disease), g in df.groupby(group_cols, dropna=False):
    rows.append({
        "state_ut": state,
        "Disease": disease,
        "rows": len(g),
        "districts": g["district"].nunique(),
        "distinct_dates": g["date"].nunique(),
        "first_date": g["date"].min(),
        "last_date": g["date"].max(),
        "cases_missing_pct": round(g["Cases"].isna().mean() * 100, 1),
        "temp_missing_pct": round(g["Temp"].isna().mean() * 100, 1),
        "preci_missing_pct": round(g["preci"].isna().mean() * 100, 1),
        "lai_missing_pct": round(g["LAI"].isna().mean() * 100, 1),
    })

summary = pd.DataFrame(rows).sort_values("rows", ascending=False)
summary.to_csv(OUT / "all_disease_state_coverage.csv", index=False)

print("Total dataset rows:", len(df))
print("States/UTs:", df["state_ut"].nunique())
print("Disease labels:", df["Disease"].nunique())
print("\n=== TOP 30 STATE-DISEASE GROUPS BY ROW COUNT ===")
print(summary.head(30).to_string(index=False))

print("\nSaved:", OUT / "all_disease_state_coverage.csv")
