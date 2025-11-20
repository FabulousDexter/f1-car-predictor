import pandas as pd
import numpy as np
from datetime import datetime


def validate_race_results(filepath):
    """
    Validate race results data quality.
    Checks for missing values, invalid data, and data integrity issues.

    Args:
        filepath: Path to the race results CSV file

    Returns:
        tuple: (is_valid, validation_report)
    """
    print("=" * 60)
    print("DATA VALIDATION - RACE RESULTS")
    print("=" * 60)

    try:
        df = pd.read_csv(filepath)
        print(f"File loaded successfully: {len(df)} rows\n")
    except FileNotFoundError:
        return False, f"File not found: {filepath}"

    issues = []
    warnings = []

    # Check 1: Required columns exist
    print("[1/8] Checking required columns...")
    required_columns = [
        "Year",
        "RoundNumber",
        "DriverNumber",
        "FullName",
        "TeamName",
        "Position",
        "SessionType",
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        print(f"Missing columns: {missing_cols}")
    else:
        print(f"All required columns present")

    # Check 2: Data types
    print("\n[2/8] Checking data types...")
    if "Year" in df.columns and not pd.api.types.is_numeric_dtype(df["Year"]):
        issues.append("Year column should be numeric")
        print("Year is not numeric")
    else:
        print("Year is numeric")

    # Check 3: Missing values in critical columns
    print("\n[3/8] Checking for missing values...")
    critical_columns = ["Year", "RoundNumber", "DriverNumber", "SessionType"]
    for col in critical_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues.append(f"{col} has {missing_count} missing values")
                print(f"{col}: {missing_count} missing values")
            else:
                print(f"{col}: No missing values")

    # Check 4: Valid year range
    print("\n[4/8] Checking year range...")
    if "Year" in df.columns:
        min_year = df["Year"].min()
        max_year = df["Year"].max()
        current_year = datetime.now().year

        if min_year < 2000 or max_year > current_year + 1:
            warnings.append(f"Unusual year range: {min_year} - {max_year}")
            print(f"Year range: {min_year} - {max_year} (check if correct)")
        else:
            print(f"Valid year range: {min_year} - {max_year}")

    # Check 5: Valid session types
    print("\n[5/8] Checking session types...")
    if "SessionType" in df.columns:
        valid_sessions = ["FP1", "FP2", "FP3", "Q", "R", "S", "SQ"]
        invalid_sessions = df[~df["SessionType"].isin(valid_sessions)][
            "SessionType"
        ].unique()
        if len(invalid_sessions) > 0:
            warnings.append(f"Unknown session types: {invalid_sessions}")
            print(f"Unknown session types: {invalid_sessions}")
        else:
            print(f"All session types are valid")

    # Check 6: Position values
    print("\n[6/8] Checking position values...")
    if "Position" in df.columns:
        race_data = df[df["SessionType"] == "R"]
        invalid_positions = race_data[
            (race_data["Position"] < 1) | (race_data["Position"] > 25)
        ]
        if len(invalid_positions) > 0:
            warnings.append(
                f"Found {len(invalid_positions)} races with unusual positions"
            )
            print(f"{len(invalid_positions)} positions outside 1-25 range")
        else:
            print(f"All race positions are in valid range")

    # Check 7: Duplicate check
    print("\n[7/8] Checking for duplicates...")
    if all(
        col in df.columns
        for col in ["Year", "RoundNumber", "SessionType", "DriverNumber"]
    ):
        duplicates = df.duplicated(
            subset=["Year", "RoundNumber", "SessionType", "DriverNumber"], keep=False
        )
        dup_count = duplicates.sum()
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate entries")
            print(f"Found {dup_count} duplicate entries")
        else:
            print(f"No duplicates found")

    # Check 8: Data completeness
    print("\n[8/8] Checking data completeness...")
    if "SessionType" in df.columns:
        session_counts = df["SessionType"].value_counts()
        print(f"  Session distribution:")
        for session, count in session_counts.items():
            print(f"    {session}: {count}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    is_valid = len(issues) == 0

    if is_valid:
        print("Data validation PASSED")
        if len(warnings) > 0:
            print(f"\{len(warnings)} warnings (non-critical):")
            for warning in warnings:
                print(f"  - {warning}")
    else:
        print(f" Data validation FAILED with {len(issues)} critical issues:")
        for issue in issues:
            print(f"  - {issue}")
        if len(warnings) > 0:
            print(f"\n  Additional {len(warnings)} warnings:")
            for warning in warnings:
                print(f"  - {warning}")

    validation_report = {
        "is_valid": is_valid,
        "total_rows": len(df),
        "critical_issues": issues,
        "warnings": warnings,
        "timestamp": datetime.now().isoformat(),
    }

    return is_valid, validation_report


def validate_lap_times(filepath):
    """
    Validate lap times data quality.

    Args:
        filepath: Path to the lap times CSV file

    Returns:
        tuple: (is_valid, validation_report)
    """
    print("\n" + "=" * 60)
    print("DATA VALIDATION - LAP TIMES")
    print("=" * 60)

    try:
        df = pd.read_csv(filepath)
        print(f"✓ File loaded successfully: {len(df)} rows\n")
    except FileNotFoundError:
        return False, f" File not found: {filepath}"

    issues = []
    warnings = []

    # Check required columns
    print("[1/3] Checking required columns...")
    required_columns = ["Year", "RoundNumber", "Driver", "LapNumber", "SessionType"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        print(f"   Missing columns: {missing_cols}")
    else:
        print(f"  ✓ All required columns present")

    # Check lap numbers
    print("\n[2/3] Checking lap numbers...")
    if "LapNumber" in df.columns:
        invalid_laps = df[(df["LapNumber"] < 1) | (df["LapNumber"] > 100)]
        if len(invalid_laps) > 0:
            warnings.append(f"Found {len(invalid_laps)} laps with unusual lap numbers")
            print(f"    {len(invalid_laps)} lap numbers outside 1-100 range")
        else:
            print(f"  ✓ All lap numbers are in valid range")

    # Check missing lap times
    print("\n[3/3] Checking lap time coverage...")
    if "LapTime" in df.columns:
        missing_laptimes = df["LapTime"].isna().sum()
        missing_pct = (missing_laptimes / len(df)) * 100
        print(f"  Missing lap times: {missing_laptimes} ({missing_pct:.1f}%)")
        if missing_pct > 30:
            warnings.append(f"High percentage of missing lap times: {missing_pct:.1f}%")
            print(f"    High missing rate (normal for outlaps/inlaps)")

    is_valid = len(issues) == 0

    print("\n" + "=" * 60)
    if is_valid:
        print(" Lap times validation PASSED")
    else:
        print(f" Lap times validation FAILED: {len(issues)} issues")
    print("=" * 60)

    validation_report = {
        "is_valid": is_valid,
        "total_rows": len(df),
        "critical_issues": issues,
        "warnings": warnings,
        "timestamp": datetime.now().isoformat(),
    }

    return is_valid, validation_report


if __name__ == "__main__":
    # Validate both datasets
    results_valid, results_report = validate_race_results(
        "data/f1_all_session_race_results.csv"
    )

    laps_valid, laps_report = validate_lap_times("data/f1_all_session_race_laps.csv")

    # Overall status
    print("\n" + "=" * 60)
    print("OVERALL VALIDATION STATUS")
    print("=" * 60)

    if results_valid and laps_valid:
        print("✅ All datasets passed validation")
        print("Ready for data cleaning and feature engineering!")
    else:
        print(" Some datasets failed validation")
        print("Please review and fix issues before proceeding")
