import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("research_outputs")
df = pd.read_csv(OUT / "mh_add_district_date.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["date", "district", "Cases"]).copy()
df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()

monthly = df.groupby("month_start").agg(
    observations=("Cases", "size"),
    districts_reporting=("district", "nunique"),
    median_cases=("Cases", "median")
)

# Reindex to every calendar month so missing months remain gaps.
calendar = pd.date_range(
    monthly.index.min(), monthly.index.max(), freq="MS"
)
monthly = monthly.reindex(calendar)
monthly.index.name = "month_start"
monthly["low_coverage"] = monthly["observations"] < 3

monthly.to_csv(OUT / "mh_add_reporting_aware_monthly.csv")

fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

axes[0].plot(monthly.index, monthly["observations"], marker=".", linestyle="none")
axes[0].set_ylabel("Records")
axes[0].set_title("Monthly recorded observations")

axes[1].plot(monthly.index, monthly["districts_reporting"],
             marker=".", linestyle="none")
axes[1].set_ylabel("Districts")
axes[1].set_title("Districts reporting per month")

axes[2].plot(monthly.index, monthly["median_cases"],
             marker=".", linestyle="none")
low = monthly["low_coverage"].fillna(False)
axes[2].scatter(
    monthly.index[low],
    monthly.loc[low, "median_cases"],
    marker="x", label="Fewer than 3 records"
)
axes[2].set_ylabel("Median cases")
axes[2].set_yscale("log")
axes[2].set_title("Median recorded cases (low-coverage months marked)")
axes[2].legend()
axes[2].set_xlabel("Month")

for ax in axes:
    ax.grid(True, alpha=0.25)

fig.suptitle("Maharashtra ADD: Temporal Patterns and Reporting Coverage")
fig.tight_layout()
fig.savefig(OUT / "mh_add_reporting_aware_timeline.png",
            dpi=180, bbox_inches="tight")

print("Saved updated timeline.")
print("Calendar months:", len(monthly))
print("Missing months:", int(monthly["observations"].isna().sum()))
print("Low-coverage observed months:", int(low.sum()))
