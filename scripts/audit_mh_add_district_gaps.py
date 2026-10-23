from pathlib import Path
import pandas as pd

SOURCE = Path("research_outputs/mh_add_district_date.csv")
OUT = Path("research_outputs")
OUT.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(
        f"{SOURCE} not found. Run scripts/audit_mh_add_cases.py first."
    )

df = pd.read_csv(SOURCE)

required = {"district", "date", "Cases"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["district", "date", "Cases"]).copy()
df["district"] = df["district"].astype(str).str.strip()

# One row per district-date is expected from the source preparation script.
duplicates = df.duplicated(["district", "date"]).sum()
print("Repeated district-date keys in input:", duplicates)

rows = []

for district, group in df.groupby("district"):
    group = group.sort_values("date")
    dates = group["date"].drop_duplicates()
    gaps = dates.diff().dt.days.dropna()

    rows.append({
        "district": district,
        "observations": len(dates),
        "first_date": dates.min().date().isoformat(),
        "last_date": dates.max().date().isoformat(),
        "span_days": (dates.max() - dates.min()).days,
        "median_gap_days": gaps.median() if len(gaps) else None,
        "mean_gap_days": gaps.mean() if len(gaps) else None,
        "max_gap_days": gaps.max() if len(gaps) else None,
        "gaps_over_30_days": int((gaps > 30).sum()),
        "gaps_over_90_days": int((gaps > 90).sum()),
    })

summary = pd.DataFrame(rows).sort_values(
    ["observations", "district"], ascending=[False, True]
)

path = OUT / "mh_add_district_gap_audit.csv"
summary.to_csv(path, index=False)

print("\n=== DISTRICT TEMPORAL GAP AUDIT ===")
print(summary.to_string(index=False))

print("\n=== COVERAGE SUMMARY ===")
print("Districts:", len(summary))
print("Districts with >= 5 observations:", int((summary["observations"] >= 5).sum()))
print("Districts with >= 10 observations:", int((summary["observations"] >= 10).sum()))
print("Districts with >= 20 observations:", int((summary["observations"] >= 20).sum()))
print("\nMedian observations per district:", summary["observations"].median())
print("Median of district median gaps (days):", summary["median_gap_days"].median())
print("Maximum gap across districts (days):", summary["max_gap_days"].max())
print("\nSaved:", path)
