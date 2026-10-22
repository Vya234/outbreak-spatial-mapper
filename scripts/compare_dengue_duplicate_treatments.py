import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("data/Final_data.csv")

for col in ["state_ut", "Disease", "district"]:
    df[col] = df[col].astype("string").str.strip()

d = df[
    (df["state_ut"] == "Maharashtra")
    & (df["Disease"] == "Dengue")
].copy()

for col in ["day", "mon", "year", "Cases"]:
    d[col] = pd.to_numeric(d[col], errors="coerce")

d["report_date"] = pd.to_datetime(
    {"year": d["year"], "month": d["mon"], "day": d["day"]},
    errors="coerce",
)
d = d.dropna(subset=["district", "report_date"]).copy()
d["month"] = d["report_date"].dt.month

# Treatment A: preserve all source rows
original = d.copy()
original["treatment"] = "All original rows"

# Treatment B: one row per district-date, median case count
dedup = (
    d.groupby(["district", "report_date"], as_index=False)
    .agg(Cases=("Cases", "median"))
)
dedup["month"] = dedup["report_date"].dt.month
dedup["treatment"] = "District-date median"

# Monthly summaries
rows = []
for name, frame in [
    ("All original rows", original),
    ("District-date median", dedup),
]:
    for month, group in frame.groupby("month"):
        rows.append({
            "treatment": name,
            "month": month,
            "observation_count": len(group),
            "median_cases": group["Cases"].median(),
            "mean_cases": group["Cases"].mean(),
        })

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "mh_dengue_duplicate_treatment_comparison.csv",
               index=False)

print("Original rows:", len(original))
print("Unique district-date records:", len(dedup))
print("\nMonthly comparison:")
print(summary.round(2).to_string(index=False))

# Compare median patterns visually
fig, ax = plt.subplots(figsize=(10, 5))

for name, group in summary.groupby("treatment"):
    group = group.sort_values("month")
    ax.plot(group["month"], group["median_cases"],
            marker="o", label=name)

ax.set_xticks(range(1, 13))
ax.set_xlabel("Calendar month")
ax.set_ylabel("Median recorded cases")
ax.set_title("Sensitivity of Monthly Dengue Pattern to Duplicate Treatment")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "mh_dengue_duplicate_sensitivity.png", dpi=200)
plt.close(fig)

print("\nSaved comparison CSV and plot in research_outputs/")
