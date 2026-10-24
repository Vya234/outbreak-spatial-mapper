
"""
HED (Global Pandemic and Epidemic Outbreaks) dataset preprocessor.

Reads the original HED Excel/CSV dataset and filters records by country
and disease. Does not generate, simulate, or interpolate records.

Default Excel sheet: "Data"
Expected columns: "Country" and "Disease"

This script is a standalone preprocessing utility. It is NOT yet
integrated into the existing EpiClim forecasting pipeline.
"""

import argparse
from pathlib import Path

import pandas as pd


COUNTRY_COL = "Country"
DISEASE_COL = "Disease"
DEFAULT_SHEET = "Data"


def read_hed_dataset(file_path: str, sheet_name: str = DEFAULT_SHEET) -> pd.DataFrame:
    """Read a HED Excel or CSV file without modifying the source."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Use .xlsx, .xls, or .csv."
        )

    # Remove accidental leading/trailing whitespace from column names.
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        raise ValueError(f"The dataset contains no rows: {path}")

    return df


def filter_hed_dataset(
    file_path: str,
    target_country: str,
    target_disease: str,
    sheet_name: str = DEFAULT_SHEET,
) -> pd.DataFrame:
    """
    Filter HED records by exact country name and case-insensitive
    disease substring. Raises an error when required columns or matches
    are missing.
    """

    df = read_hed_dataset(file_path, sheet_name)

    missing = [
        col for col in (COUNTRY_COL, DISEASE_COL)
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required column(s) {missing} not found in {file_path}.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Country uses exact matching to avoid accidental partial matches.
    country_mask = (
        df[COUNTRY_COL]
        .astype("string")
        .str.strip()
        .str.casefold()
        == target_country.strip().casefold()
    )

    # Disease uses case-insensitive substring matching.
    disease_mask = (
        df[DISEASE_COL]
        .astype("string")
        .str.contains(
            target_disease.strip(),
            case=False,
            na=False,
            regex=False,
        )
    )

    filtered = df.loc[country_mask & disease_mask].copy()

    if filtered.empty:
        available_countries = sorted(
            df[COUNTRY_COL].dropna().astype(str).str.strip().unique().tolist()
        )
        raise ValueError(
            f"No records found for disease '{target_disease}' "
            f"in country '{target_country}'.\n"
            f"Check the country/disease spelling and dataset contents.\n"
            f"Example countries in the file: {available_countries[:20]}"
        )

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Filter the HED outbreak dataset by country and disease."
    )
    parser.add_argument(
        "input_file",
        help="Path to the source HED .xlsx, .xls, or .csv file.",
    )
    parser.add_argument(
        "country",
        help='Exact country name, e.g. "India".',
    )
    parser.add_argument(
        "disease",
        help='Disease name or substring, e.g. "Dengue".',
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
        help='Excel sheet name (default: "Data"). Ignored for CSV.',
    )
    parser.add_argument(
        "--output",
        help="Optional path for saving filtered records as a CSV.",
    )

    args = parser.parse_args()

    try:
        filtered = filter_hed_dataset(
            file_path=args.input_file,
            target_country=args.country,
            target_disease=args.disease,
            sheet_name=args.sheet,
        )

        print(f"Input file: {args.input_file}")
        print(f"Country filter: {args.country}")
        print(f"Disease filter: {args.disease}")
        print(f"Matching records: {len(filtered)}")
        print(f"Columns: {list(filtered.columns)}")
        print("\nFirst matching records:")
        print(filtered.head(10).to_string(index=False))

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            filtered.to_csv(output_path, index=False)
            print(f"\nFiltered records saved to: {output_path}")

    except (FileNotFoundError, ValueError, ImportError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
