from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Paths
DATA_PATH = Path("data/Final_data.csv")
OUT_DIR = Path("research_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Clean text fields
for col in ["state_ut", "district", "Disease"]:
    df[col] = df[col].astype("string").str.strip()

# Select Maharashtra Dengue records
data = df[
    df["state_ut"].str.contains("Maharashtra", case=False, na=False)
    & df["Disease"].str.contains("Dengue", case=False, na=False)
].copy()

if data.empty:
    raise ValueError(
        "No Maharashtra Dengue rows found. Check state_ut and Disease values."
    )

# Convert numeric columns safely
for col in ["Cases", "Deaths", "Latitude", "Longitude"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Construct reporting date from day/month/year
data["report_date"] = pd.to_datetime(
    {
        "year": pd.to_numeric(data["year"], errors="coerce"),
        "month": pd.to_numeric(data["mon"], errors="coerce"),
        "day": pd.to_numeric(data["day"], errors="coerce"),
    },
    errors="coerce",
)

# Summarize each district
summary = (
    data.groupby("district", dropna=False)
    .agg(
        observation_rows=("Cases", "size"),
        distinct_reporting_dates=("report_date", "nunique"),
        total_recorded_cases=("Cases", "sum"),
        median_cases_per_observation=("Cases", "median"),
        mean_cases_per_observation=("Cases", "mean"),
        max_recorded_cases=("Cases", "max"),
        missing_case_values=("Cases", lambda s: s.isna().sum()),
        latitude=("Latitude", "median"),
        longitude=("Longitude", "median"),
    )
    .reset_index()
)

summary["total_recorded_cases"] = summary["total_recorded_cases"].round(2)
summary["median_cases_per_observation"] = (
    summary["median_cases_per_observation"].round(2)
)
summary["mean_cases_per_observation"] = (
    summary["mean_cases_per_observation"].round(2)
)

summary = summary.sort_values(
    "total_recorded_cases", ascending=False
)

summary_path = OUT_DIR / "mh_dengue_district_summary.csv"
summary.to_csv(summary_path, index=False)

# Keep districts with usable coordinates for plotting
mapped = summary.dropna(subset=["latitude", "longitude"]).copy()

def make_point_map(value_col, title, colorbar_label, filename):
    fig, ax = plt.subplots(figsize=(9, 8))

    points = ax.scatter(
        mapped["longitude"],
        mapped["latitude"],
        c=mapped[value_col],
        s=45 + 18 * mapped[value_col].fillna(0).rank(pct=True),
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
    )

    fig.colorbar(points, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    fig.text(
        0.5,
        0.015,
        "District-coordinate points; not district boundaries or population-adjusted risk.",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    output_path = OUT_DIR / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

make_point_map(
    "observation_rows",
    "Maharashtra Dengue: Reporting Coverage by District",
    "Number of dataset rows",
    "mh_dengue_observation_coverage.png",
)

make_point_map(
    "median_cases_per_observation",
    "Maharashtra Dengue: Median Recorded Cases per Observation",
    "Median recorded cases",
    "mh_dengue_median_cases_map.png",
)

# Print useful diagnostics
print("\n--- Dataset summary ---")
print(f"Filtered rows: {len(data)}")
print(f"Districts in summary: {summary['district'].nunique()}")
print(f"Rows with invalid reporting dates: {data['report_date'].isna().sum()}")
print(f"Districts with usable coordinates: {len(mapped)}")

print("\n--- Top 10 districts by total recorded cases ---")
print(
    summary[
        [
            "district",
            "observation_rows",
            "distinct_reporting_dates",
            "total_recorded_cases",
            "median_cases_per_observation",
        ]
    ].head(10).to_string(index=False)
)

print(f"\nSaved summary: {summary_path}")
