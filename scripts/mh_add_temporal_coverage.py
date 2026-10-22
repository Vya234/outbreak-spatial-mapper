import pandas as pd

path = "research_outputs/mh_add_spatial_raw.csv"
df = pd.read_csv(path)

print("Available columns:", df.columns.tolist())

# Build dates from the dataset's day, mon, and year columns
required = {"day", "mon", "year", "district", "Cases"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
for col in ["day", "mon", "year"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Date"] = pd.to_datetime(
    {"year": df["year"], "month": df["mon"], "day": df["day"]},
    errors="coerce"
)

df = df.dropna(subset=["district", "Cases", "Date"]).copy()
df["district"] = df["district"].replace({"Sholapur": "Solapur"})

# Deduplicate district-date records using the median
daily = (
    df.groupby(["district", "Date"], as_index=False)
      .agg(Cases=("Cases", "median"))
)
daily["year"] = daily["Date"].dt.year
daily["month"] = daily["Date"].dt.month

print("\nYEARLY COVERAGE")
print(
    daily.groupby("year")
         .agg(
             district_dates=("Cases", "size"),
             districts=("district", "nunique"),
             dates=("Date", "nunique"),
             first_date=("Date", "min"),
             last_date=("Date", "max")
         ).to_string()
)

print("\nMONTHLY COVERAGE: 2015 AND 2016")
subset = daily[daily["year"].isin([2015, 2016])]
print(
    subset.groupby(["year", "month"])
          .agg(
              districts=("district", "nunique"),
              district_dates=("Cases", "size"),
              dates=("Date", "nunique")
          ).to_string()
)

print("\nDISTRICT COVERAGE IN 2015 AND 2016")
print(
    subset.pivot_table(
        index="district",
        columns="year",
        values="Cases",
        aggfunc="count",
        fill_value=0
    ).to_string()
)
