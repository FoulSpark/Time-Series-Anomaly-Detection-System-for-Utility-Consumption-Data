import argparse
import pandas as pd
import sys

from src.validation.input_validator import (
    validate_input_dataset,
    print_report,
)

def main():
    parser = argparse.ArgumentParser(description="Validate input utility consumption dataset")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input dataset (CSV or Parquet)"
    )
    parser.add_argument(
        "--report-out",
        default="reports/input_validation_report.json",
        help="Path to save validation report (JSON)"
    )

    args = parser.parse_args()

    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    report = validate_input_dataset(df)

    print_report(report)
    report.save(args.report_out)

    if not report.dataset_valid:
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
