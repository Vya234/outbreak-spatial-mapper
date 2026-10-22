import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("research_outputs")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/Final_data.csv")
df = df[
    df["state_ut"].astype(str).str.strip().str.lower().eq("maharashtra")
    & df["Disease"].astype(str).str.strip().eq("Dengue")
].copy()

for col in ["Temp", "preci", "LAI", "Cases"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["month"] = pd.to_numeric(df["mon"], errors="coerce")
df = df[df["month"].between(1, 12)]

# Temperature is documented as Kelvin. Convert to Celsius for
# easier interpretation, without overwriting the source column.
df["Temp_C"] = df["Temp"] - 273.15

variables = {
    "Temp_C": "Temperature (°C)",
    "preci": "Precipitation (documented unit: mm)",
    "LAI": "Leaf Area Index (LAI)",
}

summary = df.groupby("month").agg(
    observations=("Cases", "size"),
    temperature_median=("Temp_C", "median"),
    precipitation_median=("preci", "median"),
    lai_median=("LAI", "median"),
)

print("=== MONTHLY CLIMATE SUMMARY ===")
print(summary.round(3).to_string())

summary.to_csv(OUT / "mh_dengue_climate_monthly_summary.csv")

for col, label in variables.items():
    monthly = df.groupby("month")[col].agg(
        median="median",
        observations="count",
    ).reindex(range(1, 13))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(monthly.index, monthly["median"], marker="o")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Calendar month")
    ax.set_ylabel(label)
    ax.set_title(f"Recorded {label} by Month")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filename = f"mh_dengue_{col.lower()}_by_month.png"
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)
    print("Saved:", OUT / filename)

print("Saved:", OUT / "mh_dengue_climate_monthly_summary.csv")
print("\nNote: these are summaries at disease-reporting records,")
print("not independent samples of all district-month climate conditions.")
