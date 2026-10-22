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

daily = (
    df.groupby(["district", "Date"], as_index=False)
      .agg(Cases=("Cases", "median"))
)

for year in [2015, 2016]:
    subset = daily[
        (daily["Date"].dt.year == year) &
        (daily["Date"].dt.month == 6)
    ]

    print(f"\nJUNE {year}:")
    print("District-date records:", len(subset))
    print("Distinct districts:", subset["district"].nunique())
    print("Distinct dates:", subset["Date"].nunique())

    by_date = (
        subset.groupby("Date")
              .agg(
                  districts=("district", "nunique"),
                  median_cases=("Cases", "median")
              )
              .sort_values("districts", ascending=False)
    )

    print("\nDates with the most districts reporting:")
    print(by_date.head(15).to_string())

print("\nAll June dates with at least 5 districts reporting:")
all_june = daily[daily["Date"].dt.month == 6]
counts = all_june.groupby("Date")["district"].nunique()
print(counts[counts >= 5].sort_values(ascending=False).to_string())
