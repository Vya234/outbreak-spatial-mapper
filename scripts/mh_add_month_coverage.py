import pandas as pd

df = pd.read_csv("research_outputs/mh_add_spatial_raw.csv")

for col in ["day", "mon", "year", "Cases"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Date"] = pd.to_datetime(
    {"year": df["year"], "month": df["mon"], "day": df["day"]},
    errors="coerce"
)

df = df.dropna(subset=["district", "Cases", "Date"]).copy()
df["district"] = df["district"].replace({"Sholapur": "Solapur"})

# Deduplicate district-date observations
daily = (
    df.groupby(["district", "Date"], as_index=False)
      .agg(Cases=("Cases", "median"))
)
daily["month"] = daily["Date"].dt.month
daily["year"] = daily["Date"].dt.year

print("COVERAGE BY CALENDAR MONTH, ALL YEARS")
print(
    daily.groupby("month")
         .agg(
             districts=("district", "nunique"),
             district_dates=("Cases", "size"),
             years=("year", "nunique")
         ).to_string()
)

print("\nCOVERAGE BY MONTH AND YEAR")
print(
    daily.pivot_table(
        index="month",
        columns="year",
        values="district",
        aggfunc="nunique",
        fill_value=0
    ).to_string()
)
