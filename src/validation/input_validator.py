from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json
import os

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "meter_id",
    "date",
    "consumption_kwh",
]

OPTIONAL_COLUMNS = [
    "region_id",
    "customer_type",
    "temperature_c",
    "is_holiday",
    "data_source",
    "generated_anomaly",  
]


@dataclass
class ValidationReport:
    dataset_valid: bool
    blocking_issues: List[str]
    warnings: List[str]
    summary: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def _add_issue(issues: List[str], msg: str) -> None:
    issues.append(msg)


def validate_input_dataset(
    df: pd.DataFrame,
    *,
    min_days_per_meter: int = 30,
    max_missing_ratio_per_meter: float = 0.5,
    max_flatline_run: int = 14,
    expected_unit_hint: str = "kWh",
) -> ValidationReport:
    """
    Validates whether a user-provided dataset is acceptable for scoring.

    Blocking issues => dataset_valid = False (stop pipeline).
    Warnings => dataset_valid remains True (continue, but flagged).

    Assumptions:
    - daily granularity (per meter)
    - 1 row per (meter_id, date)
    - consumption_kwh numeric and non-negative
    """

    blocking: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {}


    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        _add_issue(blocking, f"Missing required columns: {missing_cols}")
        return ValidationReport(False, blocking, warnings, {"reason": "missing_required_columns"})

    extra_cols = [c for c in df.columns if c not in (REQUIRED_COLUMNS + OPTIONAL_COLUMNS)]
    if extra_cols:
        _add_issue(warnings, f"Extra columns will be ignored: {extra_cols}")


    if df["meter_id"].isna().any():
        _add_issue(blocking, "Found null meter_id values.")
    else:
    
        df["meter_id"] = df["meter_id"].astype(str)

   
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except Exception as e:
        _add_issue(blocking, f"Failed to parse 'date' column to datetime: {e}")

    df["consumption_kwh"] = pd.to_numeric(df["consumption_kwh"], errors="coerce")
    if df["consumption_kwh"].isna().any():
        bad = int(df["consumption_kwh"].isna().sum())
        _add_issue(blocking, f"'consumption_kwh' has {bad} non-numeric / NaN values.")
    if (df["consumption_kwh"] < 0).any():
        bad = int((df["consumption_kwh"] < 0).sum())
        _add_issue(blocking, f"'consumption_kwh' has {bad} negative values (not allowed).")

    if blocking:
        return ValidationReport(False, blocking, warnings, {"reason": "blocking_type_or_parse_errors"})

   
    dup_mask = df.duplicated(subset=["meter_id", "date"], keep=False)
    if dup_mask.any():
        dup_count = int(dup_mask.sum())
        sample = df.loc[dup_mask, ["meter_id", "date"]].head(5).to_dict("records")
        _add_issue(blocking, f"Duplicate (meter_id, date) rows found: {dup_count}. Example: {sample}")

    if blocking:
        return ValidationReport(False, blocking, warnings, {"reason": "duplicate_keys"})

   
    df_sorted = df.sort_values(["meter_id", "date"]).copy()
    meters = df_sorted["meter_id"].nunique()
    rows = len(df_sorted)
    date_min = df_sorted["date"].min()
    date_max = df_sorted["date"].max()

    summary.update({
        "rows": rows,
        "meters": meters,
        "date_min": date_min,
        "date_max": date_max,
        "expected_unit_hint": expected_unit_hint,
    })


    per_meter_counts = df_sorted.groupby("meter_id")["date"].nunique()
    low_days = per_meter_counts[per_meter_counts < min_days_per_meter]
    if len(low_days) > 0:
        _add_issue(blocking, f"{len(low_days)} meters have < {min_days_per_meter} days of data (insufficient history). "
                             f"Example meters: {list(low_days.index[:5])}")


    def missing_ratio_for_meter(mdf: pd.DataFrame) -> float:
        dmin = mdf["date"].min()
        dmax = mdf["date"].max()
        expected = (dmax - dmin).days + 1
        observed = mdf["date"].nunique()
        if expected <= 0:
            return 1.0
        return 1.0 - (observed / expected)

    missing_ratios = df_sorted.groupby("meter_id", group_keys=False).apply(missing_ratio_for_meter)
    very_sparse = missing_ratios[missing_ratios > max_missing_ratio_per_meter]
    if len(very_sparse) > 0:
        _add_issue(warnings, f"{len(very_sparse)} meters are very sparse (> {max_missing_ratio_per_meter:.0%} missing days). "
                             f"Example meters: {list(very_sparse.index[:5])}")

    if blocking:
        return ValidationReport(False, blocking, warnings, {**summary, "reason": "insufficient_history"})

    
    def max_flatline_run_len(values: pd.Series) -> int:
    
        v = values.to_numpy()
        if len(v) == 0:
            return 0
        run = 1
        best = 1
        for i in range(1, len(v)):
            if v[i] == v[i - 1]:
                run += 1
                best = max(best, run)
            else:
                run = 1
        return best

    flatline = df_sorted.groupby("meter_id")["consumption_kwh"].apply(max_flatline_run_len)
    bad_flatline = flatline[flatline >= max_flatline_run]
    if len(bad_flatline) > 0:
        _add_issue(warnings, f"{len(bad_flatline)} meters have flatline runs >= {max_flatline_run} days (possible stuck meter/ETL issue). "
                             f"Example meters: {list(bad_flatline.index[:5])}")

   
    p50 = float(df_sorted["consumption_kwh"].quantile(0.50))
    p99 = float(df_sorted["consumption_kwh"].quantile(0.99))
    summary.update({"p50_consumption_kwh": p50, "p99_consumption_kwh": p99})

    if p99 > 100000:
        _add_issue(warnings, "Consumption appears extremely high (p99 > 100,000 kWh). Possible unit mismatch or industrial aggregation.")
    if p50 < 0.01:
        _add_issue(warnings, "Consumption appears extremely low (median < 0.01 kWh). Possible unit mismatch (Wh vs kWh).")

    dataset_valid = len(blocking) == 0
    return ValidationReport(dataset_valid, blocking, warnings, summary)


def print_report(report: ValidationReport) -> None:
    """Human-readable console output."""
    if report.dataset_valid:
        print("Input validation: PASSED ✅")
    else:
        print("Input validation: FAILED ❌")

    if report.blocking_issues:
        print("\nBlocking issues:")
        for i, msg in enumerate(report.blocking_issues, start=1):
            print(f"  {i}. {msg}")

    if report.warnings:
        print("\nWarnings:")
        for i, msg in enumerate(report.warnings, start=1):
            print(f"  {i}. {msg}")

    print("\nSummary:")
    for k, v in report.summary.items():
        print(f"  - {k}: {v}")
