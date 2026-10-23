
import pandas as pd
from pathlib import Path

DATA = "data/Final_data.csv"

# Records manually checked against the uploaded official IDSP PDFs.
verified = [
    {
        "state": "Maharashtra",
        "district": "Parbhani",
        "disease": "Acute Diarrheal Disease",
        "date": "2020-10-05",
        "cases": 66,
        "bulletin_week": 53,
        "reporting_date": "2020-12-31",
        "source": "week53_2020.pdf",
        "page": 4,
    },
    {
        "state": "Maharashtra",
        "district": "Kolhapur",
        "disease": "Acute Diarrheal Disease",
        "date": "2020-12-07",
        "cases": 72,
        "bulletin_week": 52,
        "reporting_date": None,
        "source": "week52_2020.pdf",
        "page": 4,
    },
    {
        "state": "Maharashtra",
        "district": "Satara",
        "disease": "Acute Diarrheal Disease",
        "date": "2020-12-07",
        "cases": 116,
        "bulletin_week": 52,
        "reporting_date": None,
        "source": "week52_2020.pdf",
        "page": 4,
    },
]

df = pd.read_csv(DATA)

df["date"] = pd.to_datetime(
    dict(year=df["year"], month=df["mon"], day=df["day"]),
    errors="coerce",
).dt.strftime("%Y-%m-%d")

def norm(value):
    return str(value).strip().lower()

results = []

for item in verified:
    mask = (
        df["state_ut"].map(norm).eq(norm(item["state"]))
        & df["district"].map(norm).eq(norm(item["district"]))
        & df["Disease"].map(norm).str.contains(
            "diarr", na=False
        )
        & df["date"].eq(item["date"])
        & pd.to_numeric(df["Cases"], errors="coerce").eq(
            item["cases"]
        )
    )

    matches = df.loc[mask]

    results.append({
        **item,
        "dataset_matches": len(matches),
        "dataset_weeks": "; ".join(
            matches["week_of_outbreak"].astype(str).unique()
        ),
        "status": (
            "MATCH"
            if len(matches) >= 1
            else "NOT FOUND IN DATASET"
        ),
    })

out = pd.DataFrame(results)
Path("research_outputs").mkdir(exist_ok=True)
out.to_csv(
    "research_outputs/idsp_validation.csv",
    index=False,
)

print(out[
    [
        "district", "date", "cases", "bulletin_week",
        "dataset_matches", "dataset_weeks", "status"
    ]
].to_string(index=False))

print("\nSaved: research_outputs/idsp_validation.csv")
