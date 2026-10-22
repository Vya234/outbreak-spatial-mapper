from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path("research_outputs")
OUT.mkdir(exist_ok=True)

# Load district-date data already deduplicated by district and date
df = pd.read_csv(OUT / "mh_add_district_date.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["district", "date", "Cases"]).copy()

df["month"] = df["date"].dt.month

# Each cell is the median recorded case count for that district
# in that calendar month, pooling all available years.
heatmap_data = (
    df.groupby(["district", "month"])["Cases"]
      .median()
      .unstack("month")
      .reindex(columns=range(1, 13))
      .sort_index()
)

heatmap_data.columns = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

heatmap_data.to_csv(OUT / "mh_add_district_month_heatmap.csv")

fig, ax = plt.subplots(figsize=(14, 11))

sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    mask=heatmap_data.isna(),
    linewidths=0.25,
    linecolor="white",
    cbar_kws={"label": "Median recorded ADD cases"},
    ax=ax,
)

ax.set_title(
    "Maharashtra Acute Diarrhoeal Disease\n"
    "District-wise Median Recorded Cases by Calendar Month "
    "(All Available Years)"
)
ax.set_xlabel("Calendar month")
ax.set_ylabel("District")
ax.tick_params(axis="y", labelsize=8)

fig.tight_layout()

image_path = OUT / "mh_add_district_month_heatmap.png"
fig.savefig(image_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Districts:", heatmap_data.shape[0])
print("Months:", heatmap_data.shape[1])
print("Observed district-month cells:", int(heatmap_data.notna().sum().sum()))
print("Missing district-month cells:", int(heatmap_data.isna().sum().sum()))
print("\nSaved:")
print(" ", OUT / "mh_add_district_month_heatmap.csv")
print(" ", image_path)
print(
    "\nInterpretation caution: months pool observations across years. "
    "Blank cells mean no recorded observations, not zero cases. "
    "Unequal reporting coverage prevents direct conclusions about "
    "seasonality or disease incidence."
)
