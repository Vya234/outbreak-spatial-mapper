import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

DATA = "research_outputs/mh_add_spatial_raw.csv"
BOUNDARIES = "data/boundaries/maharashtra_districts.geojson"
OUT = "research_outputs"

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(
    dict(year=df.year, month=df.mon, day=df.day),
    errors="coerce"
)
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna(subset=["date", "district", "Cases"]).copy()
df["district_key"] = (
    df["district"].str.strip().replace({"Sholapur": "Solapur"})
)

df["period"] = pd.cut(
    df["date"].dt.year,
    bins=[2008, 2013, 2018, 2022],
    labels=["2009-2013", "2014-2018", "2019-2022"]
)

summary = df.groupby(
    ["district_key", "period"], observed=True
).agg(
    median_cases=("Cases", "median"),
    observations=("Cases", "size")
).reset_index()

summary.to_csv(f"{OUT}/mh_add_spatiotemporal_summary.csv", index=False)

polygons = gpd.read_file(BOUNDARIES)
name_col = next(
    (c for c in polygons.columns
     if c.lower() in ["district", "district_name", "dtname", "name"]),
    None
)
if name_col is None:
    raise ValueError(f"Cannot identify district name column: {polygons.columns.tolist()}")

polygons["district_key"] = (
    polygons[name_col].astype(str).str.strip()
    .replace({"Sholapur": "Solapur"})
)

periods = ["2009-2013", "2014-2018", "2019-2022"]
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
vmin = summary["median_cases"].min()
vmax = summary["median_cases"].max()
norm = Normalize(vmin=vmin, vmax=vmax)

for ax, period in zip(axes, periods):
    part = summary[summary["period"] == period]
    geo = polygons.merge(
        part, on="district_key", how="left"
    )

    geo.plot(
        column="median_cases",
        ax=ax,
        cmap="YlOrRd",
        norm=norm,
        legend=True,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
            "label": "No observations"
        },
        legend_kwds={"shrink": 0.65}
    )
    ax.set_title(
        f"{period}\n"
        f"Districts observed: {part['district_key'].nunique()}"
    )
    ax.axis("off")

fig.suptitle(
    "Maharashtra Acute Diarrhoeal Disease: "
    "Median Recorded Cases by Period",
    fontsize=14
)
plt.tight_layout()
plt.savefig(f"{OUT}/mh_add_spatiotemporal_maps.png",
            dpi=200, bbox_inches="tight")
print("Saved map: research_outputs/mh_add_spatiotemporal_maps.png")
print("Saved summary: research_outputs/mh_add_spatiotemporal_summary.csv")
print("\nCoverage:")
print(summary.groupby("period", observed=True).agg(
    districts=("district_key", "nunique"),
    observations=("observations", "sum")
).to_string())
