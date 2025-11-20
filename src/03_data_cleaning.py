import pandas as pd
import numpy as np
from datetime import datetime
import pytz


def filter_completed_races(df):
    """
    Filter to only completed races (past events).
    For current year, only include races where Session5Date (race day) has passed.

    Args:
        df: Raw race results DataFrame

    Returns:
        DataFrame: Only completed races
    """
    print("\n[Pre-filter] Filtering completed races only...")
    initial_count = len(df)

    # Get current time in UTC
    now_utc = datetime.now(pytz.utc)

    # For race sessions (SessionType == 'R'), check if race has occurred
    # Assume races without a Year column are invalid
    if "Year" not in df.columns:
        print("  Warning: No 'Year' column found, skipping date filtering")
        return df

    current_year = now_utc.year

    # Keep all past years completely
    past_years = df[df["Year"] < current_year].copy()

    # For current year, need to check race dates if available
    current_year_races = df[df["Year"] == current_year].copy()

    # Try to load event schedule to get race dates
    try:
        import fastf1

        schedule = fastf1.get_event_schedule(current_year, include_testing=False)
        schedule["Session5DateUtc"] = pd.to_datetime(
            schedule["Session5DateUtc"]
        ).dt.tz_localize("UTC")

        # Get completed round numbers
        completed_rounds = schedule[schedule["Session5DateUtc"] < now_utc][
            "RoundNumber"
        ].tolist()

        # Filter current year to only completed rounds
        completed_current = current_year_races[
            current_year_races["RoundNumber"].isin(completed_rounds)
        ].copy()

        print(
            f"  Current year ({current_year}): {len(completed_current)} completed races out of {len(current_year_races)}"
        )
    except Exception as e:
        print(f"  Warning: Could not fetch schedule for date filtering: {e}")
        print(f"  Keeping all {current_year} races")
        completed_current = current_year_races

    # Combine past years + completed current year
    df_filtered = pd.concat([past_years, completed_current], ignore_index=True)

    removed = initial_count - len(df_filtered)
    print(
        f"  Filtered to {len(df_filtered)} completed races (removed {removed} future/incomplete)"
    )

    return df_filtered


def clean_race_results(input_file, output_file):
    """
    Clean race results data:
    - Handle missing values
    - Standardize team names
    - Fix data types
    - Remove invalid entries

    Args:
        input_file: Path to raw race results CSV
        output_file: Path to save cleaned CSV
    """
    print("=" * 60)
    print("DATA CLEANING - RACE RESULTS")
    print("=" * 60)

    # Load data
    df = pd.read_csv(input_file)
    print(f"\nLoaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    initial_rows = len(df)

    # Step 0: Filter to completed races only (removes future races)
    df = filter_completed_races(df)

    # Step 1: Filter race sessions only
    print("\n[1/7] Filtering race sessions...")
    df = df[df["SessionType"] == "R"].copy()
    print(f"  Kept {len(df)} race rows (removed {initial_rows - len(df)} non-race)")

    # Step 2: Handle missing Position values
    print("\n[2/7] Handling missing positions...")
    missing_positions = df["Position"].isna().sum()
    if missing_positions > 0:
        print(f"  Removing {missing_positions} rows with missing positions (DNF/DNS)")
        df = df[df["Position"].notna()]

    # Step 3: Convert data types
    print("\n[3/7] Converting data types...")
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df.dropna(subset=["Position"])
    df["Position"] = df["Position"].astype(int)

    df["Year"] = df["Year"].astype(int)
    df["RoundNumber"] = df["RoundNumber"].astype(int)
    df["DriverNumber"] = df["DriverNumber"].astype(int)

    if "GridPosition" in df.columns:
        df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")

    if "Points" in df.columns:
        df["Points"] = pd.to_numeric(df["Points"], errors="coerce").fillna(0)

    print(f"   Data types converted")

    # Step 4: Standardize team names
    print("\n[4/7] Standardizing team names...")
    team_mapping = {
        # Red Bull family
        "Red Bull": "Red Bull Racing",
        "Red Bull Racing": "Red Bull Racing",
        # RB/AlphaTauri/Toro Rosso evolution
        "RB": "RB F1 Team",
        "Racing Bulls": "RB F1 Team",
        "AlphaTauri": "RB F1 Team",
        "Toro Rosso": "RB F1 Team",
        # Alpine
        "Alpine F1 Team": "Alpine",
        "Alpine": "Alpine",
        # Sauber family
        "Kick Sauber": "Sauber",
        "Alfa Romeo": "Sauber",
        "Sauber": "Sauber",
        # Other teams - standardize
        "Haas F1 Team": "Haas",
        "Haas": "Haas",
        "Aston Martin": "Aston Martin",
        "Ferrari": "Ferrari",
        "McLaren": "McLaren",
        "Mercedes": "Mercedes",
        "Williams": "Williams",
    }

    df["TeamName"] = df["TeamName"].replace(team_mapping)
    print(f"  Unique teams: {df['TeamName'].nunique()}")
    print(f"  Teams: {sorted(df['TeamName'].unique())}")

    # Step 5: Handle grid position anomalies
    print("\n[5/7] Handling grid positions...")
    if "GridPosition" in df.columns:
        # Pit lane starts are 0.0, replace with max grid + 1
        max_grid = df["GridPosition"].max()
        pit_starts = (df["GridPosition"] == 0.0).sum()
        if pit_starts > 0:
            df["GridPosition"] = df["GridPosition"].replace(0.0, max_grid + 1)
            print(f"  Fixed {pit_starts} pit lane starts (0 → {max_grid + 1})")

        # Fill remaining NaN with max + 1
        missing_grids = df["GridPosition"].isna().sum()
        if missing_grids > 0:
            df["GridPosition"] = df["GridPosition"].fillna(max_grid + 1)
            print(f"  Filled {missing_grids} missing grid positions")

    # Step 6: Remove duplicates
    print("\n[6/7] Removing duplicates...")
    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=["Year", "RoundNumber", "SessionType", "DriverNumber"]
    )
    removed_dupes = before_dedup - len(df)
    if removed_dupes > 0:
        print(f"  Removed {removed_dupes} duplicate entries")
    else:
        print(f"  No duplicates found")

    # Step 7: Sort data
    print("\n[7/7] Sorting data...")
    df = df.sort_values(["Year", "RoundNumber", "Position"])
    df = df.reset_index(drop=True)
    print(f"  ✓ Data sorted by Year, Round, Position")

    # Save cleaned data
    df.to_csv(output_file, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Input rows:  {initial_rows}")
    print(f"Output rows: {len(df)}")
    print(
        f"Removed:     {initial_rows - len(df)} ({((initial_rows - len(df))/initial_rows*100):.1f}%)"
    )
    print(f"\n Cleaned data saved to: {output_file}")

    return df


def clean_lap_times(input_file, output_file):
    """
    Clean lap times data:
    - Convert lap times to seconds
    - Remove outliers
    - Handle missing values

    Args:
        input_file: Path to raw lap times CSV
        output_file: Path to save cleaned CSV
    """
    print("\n" + "=" * 60)
    print("DATA CLEANING - LAP TIMES")
    print("=" * 60)

    df = pd.read_csv(input_file)
    print(f"\nLoaded {len(df)} rows")

    initial_rows = len(df)

    # Step 1: Filter race laps only
    print("\n[1/5] Filtering race laps...")
    df = df[df["SessionType"] == "R"].copy()
    print(f"  Kept {len(df)} race laps")

    # Step 2: Convert lap time to seconds
    print("\n[2/5] Converting lap times to seconds...")

    def laptime_to_seconds(laptime):
        """Convert timedelta or string laptime to seconds"""
        if pd.isna(laptime):
            return None
        try:
            if isinstance(laptime, str):
                if "days" in laptime:
                    parts = laptime.split()
                    time_part = parts[-1]
                else:
                    time_part = laptime
                h, m, s = time_part.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            else:
                # Assume it's already numeric (seconds)
                return float(laptime)
        except:
            return None

    df["LapTimeSeconds"] = df["LapTime"].apply(laptime_to_seconds)
    converted = df["LapTimeSeconds"].notna().sum()
    print(f"  Converted {converted} lap times to seconds")

    # Step 3: Remove outliers (too fast or too slow)
    print("\n[3/5] Removing outlier lap times...")
    valid_laps = df[
        (df["LapTimeSeconds"] >= 70)  # Faster than 1:10 (unrealistic)
        & (df["LapTimeSeconds"] <= 150)  # Slower than 2:30 (safety car pace)
    ]
    outliers_removed = len(df) - len(valid_laps)
    df = valid_laps.copy()
    print(f"  Removed {outliers_removed} outlier laps")
    print(
        f"  Valid lap time range: {df['LapTimeSeconds'].min():.2f}s - {df['LapTimeSeconds'].max():.2f}s"
    )

    # Step 4: Clean compound names
    print("\n[4/5] Cleaning tyre compound names...")
    if "Compound" in df.columns:
        df["Compound"] = (
            df["Compound"].str.upper().replace({"": "UNKNOWN", "None": "UNKNOWN"})
        )
        compounds = df["Compound"].value_counts()
        print(f"  Compound distribution:")
        for compound, count in compounds.items():
            print(f"    {compound}: {count}")

    # Step 5: Remove duplicates
    print("\n[5/5] Removing duplicates...")
    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=["Year", "RoundNumber", "SessionType", "Driver", "LapNumber"]
    )
    removed_dupes = before_dedup - len(df)
    if removed_dupes > 0:
        print(f"  Removed {removed_dupes} duplicate laps")
    else:
        print(f"  No duplicates found")

    # Sort and save
    df = df.sort_values(["Year", "RoundNumber", "Driver", "LapNumber"])
    df = df.reset_index(drop=True)
    df.to_csv(output_file, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Input rows:  {initial_rows}")
    print(f"Output rows: {len(df)}")
    print(
        f"Removed:     {initial_rows - len(df)} ({((initial_rows - len(df))/initial_rows*100):.1f}%)"
    )
    print(f"\n Cleaned lap times saved to: {output_file}")

    return df


if __name__ == "__main__":
    # Clean both datasets
    print("Starting data cleaning pipeline...\n")

    clean_race_results(
        input_file="data/f1_all_session_race_results.csv",
        output_file="data/f1_all_session_race_results_clean.csv",
    )

    clean_lap_times(
        input_file="data/f1_all_session_race_laps.csv",
        output_file="data/f1_all_session_race_laps_clean.csv",
    )

    print("\n" + "=" * 60)
    print(" DATA CLEANING COMPLETE!")
    print("=" * 60)
    print("\nNext step: Run feature engineering")
    print("  python src/04_feature_engineering.py")
