
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "state_ut", "district", "Disease", "Cases", "Deaths",
    "day", "mon", "year", "Latitude", "Longitude"
}

ENVIRONMENT_COLUMNS = ["preci", "LAI", "Temp"]


class EpiClimAdapter:
    def __init__(self, filepath="Final_data.csv"):
        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Source CSV not found: {self.filepath}"
            )

        self.df = pd.read_csv(self.filepath)

        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

    def filter_observations(
        self,
        state="Maharashtra",
        disease="Dengue",
        include_mixed_labels=False
    ):
        df = self.df.copy()

        # Normalize text without changing the source CSV.
        for col in ["state_ut", "district", "Disease"]:
            df[col] = df[col].astype("string").str.strip()

        df = df[
            df["state_ut"].str.casefold()
            == state.strip().casefold()
        ]

        if include_mixed_labels:
            df = df[
                df["Disease"].str.contains(
                    disease, case=False, na=False
                )
            ]
        else:
            df = df[
                df["Disease"].str.casefold()
                == disease.strip().casefold()
            ]

        if df.empty:
            raise ValueError(
                f"No records found for {state} / {disease}"
            )

        # Construct actual calendar dates.
        df["Date"] = pd.to_datetime(
            {
                "year": df["year"],
                "month": df["mon"],
                "day": df["day"]
            },
            errors="coerce"
        )

        # Convert numeric fields robustly.
        for col in [
            "Cases", "Deaths", "Latitude", "Longitude",
            *ENVIRONMENT_COLUMNS
        ]:
            if col in df.columns:
                cleaned = (
                    df[col]
                    .astype("string")
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                df[col] = pd.to_numeric(
                    cleaned, errors="coerce"
                )

        # Invalid dates, coordinates and case counts are excluded.
        df = df.dropna(
            subset=["Date", "Latitude", "Longitude", "Cases"]
        ).copy()

        df = df[
            df["Cases"].ge(0)
            & df["Latitude"].between(-90, 90)
            & df["Longitude"].between(-180, 180)
        ].copy()

        # Preserve the original observations; do not create
        # artificial case points or artificial daily timestamps.
        columns = [
            "Date", "state_ut", "district", "Disease",
            "Cases", "Deaths", "Latitude", "Longitude"
        ]
        columns += [
            c for c in ENVIRONMENT_COLUMNS if c in df.columns
        ]

        return (
            df[columns]
            .sort_values(["Date", "district"])
            .reset_index(drop=True)
        )

    @staticmethod
    def aggregate_observed_dates(observations):
        """Summarize records on dates present in the source."""
        aggregations = {
            "Cases": ("Cases", "sum"),
            "district_count": ("district", "nunique"),
            "record_count": ("district", "size"),
            "centroid_lat": ("Latitude", "mean"),
            "centroid_lon": ("Longitude", "mean")
        }

        if "Deaths" in observations.columns:
            aggregations["Deaths"] = ("Deaths", "sum")

        for col in ENVIRONMENT_COLUMNS:
            if col in observations.columns:
                aggregations[f"mean_{col}"] = (col, "mean")

        daily = (
            observations.groupby("Date", as_index=False)
            .agg(**aggregations)
            .sort_values("Date")
            .reset_index(drop=True)
        )

        daily["elapsed_days_from_start"] = (
            daily["Date"] - daily["Date"].min()
        ).dt.days

        # Missing dates remain missing. They are not assumed to
        # represent zero reported cases.
        return daily

    def export(self, output_dir="data_corrected"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        observations = self.filter_observations()
        daily = self.aggregate_observed_dates(observations)

        observations_path = (
            output_dir / "epiclim_mh_dengue_observations.csv"
        )
        daily_path = (
            output_dir / "epiclim_mh_dengue_by_observed_date.csv"
        )

        observations.to_csv(
            observations_path, index=False, date_format="%Y-%m-%d"
        )
        daily.to_csv(
            daily_path, index=False, date_format="%Y-%m-%d"
        )

        print("Source:", self.filepath)
        print("Observation records:", len(observations))
        print("Distinct dates:", observations["Date"].nunique())
        print("Districts:", observations["district"].nunique())
        print(
            "Date range:",
            observations["Date"].min().date(),
            "to",
            observations["Date"].max().date()
        )
        print("Observation CSV:", observations_path)
        print("Date summary CSV:", daily_path)

        return observations, daily


if __name__ == "__main__":
    adapter = EpiClimAdapter("data/Final_data.csv")
    observations, daily = adapter.export()
