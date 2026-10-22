from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "Final_data.csv"
OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Load the original dataset without modifying it.
df = pd.read_csv(SOURCE)
df["Source_Row_ID"] = df["Unnamed: 0"]

# Select one disease-region pair for the initial case study.
df = df[
    df["state_ut"].astype(str).str.strip().eq("Maharashtra")
    & df["Disease"].astype(str).str.strip().eq(
        "Acute Diarrhoeal Disease"
    )
].copy()

# Construct the recorded calendar date.
df["Date"] = pd.to_datetime(
    dict(
        year=pd.to_numeric(df["year"], errors="coerce"),
        month=pd.to_numeric(df["mon"], errors="coerce"),
        day=pd.to_numeric(df["day"], errors="coerce"),
    ),
    errors="coerce",
)

for col in ["Cases", "Deaths", "Temp", "preci", "LAI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Flag exact duplicates, excluding the source row ID.
duplicate_cols = [
    c for c in df.columns
    if c not in {"Unnamed: 0", "Source_Row_ID", "Exact_Duplicate"}
]

df["Exact_Duplicate"] = df.duplicated(
    subset=duplicate_cols,
    keep=False,
)

# Preserve every source observation in the audit file.
df.to_csv(OUT / "maharashtra_add_audit.csv", index=False)

# Create a separate descriptive table.
valid = df.dropna(subset=["Date", "Cases"]).copy()

district_summary = (
    valid.groupby("district")
    .agg(
        records=("Cases", "size"),
        distinct_dates=("Date", "nunique"),
        total_recorded_cases=("Cases", "sum"),
        median_recorded_cases=("Cases", "median"),
    )
    .sort_values("records", ascending=False)
)

district_summary.to_csv(
    OUT / "maharashtra_add_district_summary.csv"
)

# Visualization 1: reported cases by recorded date.
# These are observations, not a complete daily incidence series.
plt.figure(figsize=(12, 5))
plt.scatter(valid["Date"], valid["Cases"], alpha=0.7)
plt.xlabel("Recorded date")
plt.ylabel("Reported cases")
plt.title("Maharashtra ADD: Recorded Case Observations")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "maharashtra_add_observations.png", dpi=200)
plt.close()

# Visualization 2: total recorded cases by district.
plt.figure(figsize=(10, 8))
district_summary["total_recorded_cases"].sort_values().plot(
    kind="barh"
)
plt.xlabel("Sum of recorded case counts")
plt.ylabel("District")
plt.title("Maharashtra ADD: Recorded Cases by District")
plt.tight_layout()
plt.savefig(OUT / "maharashtra_add_districts.png", dpi=200)
plt.close()

# Climate coverage and duplicate summary.
climate_cols = ["Temp", "preci", "LAI"]

report = [
    "EpiClim Maharashtra ADD analysis",
    f"Source file: {SOURCE}",
    f"Rows selected: {len(df)}",
    f"Valid date/case observations: {len(valid)}",
    f"Districts: {valid['district'].nunique()}",
    f"Date range: {valid['Date'].min()} to {valid['Date'].max()}",
    f"Rows flagged as exact-duplicate candidates: {int(df['Exact_Duplicate'].sum())}",
    "",
    "Missing values:",
    df[["Cases", *climate_cols]].isna().sum().to_string(),
    "",
    "Important limitation:",
    "Recorded dates and week labels have not been reconciled.",
    "No missing reporting period is assumed to represent zero cases.",
    "Duplicate candidates are flagged, not automatically removed.",
]

(OUT / "analysis_notes.txt").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("\n".join(report))
print("\nGenerated files:")
for path in sorted(OUT.iterdir()):
    if path.is_file():
        print("-", path.relative_to(ROOT))
