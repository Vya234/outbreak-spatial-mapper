import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(OUT / "mh_add_district_date.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["district", "date", "Cases"]).copy()

summary = (
    df.groupby("district")
    .agg(
        observations=("Cases", "size"),
        median_recorded_cases=("Cases", "median"),
        mean_recorded_cases=("Cases", "mean"),
        max_recorded_cases=("Cases", "max"),
        sum_recorded_cases=("Cases", "sum"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    .reset_index()
)

summary = summary.sort_values("median_recorded_cases", ascending=False)
summary.to_csv(OUT / "mh_add_district_summary.csv", index=False)

print("=== TOP 15 DISTRICTS BY MEDIAN RECORDED CASES ===")
print(summary.head(15).round(2).to_string(index=False))

print("\n=== TOP 15 DISTRICTS BY SUM OF RECORDED CASES ===")
print(
    summary.sort_values("sum_recorded_cases", ascending=False)
    .head(15)
    .round(2)
    .to_string(index=False)
)

# Plot the districts with the largest medians.
plot_data = summary.head(15).sort_values("median_recorded_cases")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(plot_data["district"], plot_data["median_recorded_cases"])
ax.set_xlabel("Median recorded cases per observed district-date")
ax.set_ylabel("District")
ax.set_title("Maharashtra ADD: District-Level Recorded Case Counts")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()

path = OUT / "mh_add_district_median_cases.png"
fig.savefig(path, dpi=160)
plt.close(fig)

print("\nSaved:", OUT / "mh_add_district_summary.csv")
print("Saved:", path)
print(
    "\nInterpretation caution: totals and medians depend on reporting "
    "coverage and are not population-normalized incidence rates."
)
