import pandas as pd
import numpy as np


def load_raw_data(filepath):
    """Load cleaned race results data"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df


def create_target_variable(df):
    """Create binary target: IsTop3 (1 if Position <= 3, else 0)"""
    df["IsTop3"] = (df["Position"] <= 3).astype(int)
    print(f"Created target variable. Top 3 finishes: {df['IsTop3'].sum()}")
    return df


def calculate_recent_form(df, window=3):
    """
    Calculate rolling statistics for recent performance:
    - recent_avg_position: Average finishing position in last N races
    - recent_podiums: Number of podiums in last N races
    - position_consistency: Standard deviation of recent positions
    """
    df["recent_avg_position"] = df.groupby("DriverNumber")["Position"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df["recent_podiums"] = df.groupby("DriverNumber")["IsTop3"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    df["position_consistency"] = df.groupby("DriverNumber")["Position"].transform(
        lambda x: x.rolling(5, min_periods=1).std().fillna(10)
    )
    return df


def calculate_cumulative_points(df):
    """
    Calculate points accumulated before each race:
    - driver_points_before_race
    - team_points_before_race
    - points_per_race: Driver points per race average
    - team_avg_position: Team average position in each year
    """
    df = df.sort_values(by=["Year", "RoundNumber"])
    df["driver_points_before_race"] = (
        df.groupby("DriverNumber")["Points"].cumsum().shift(1).fillna(0)
    )
    df["team_points_before_race"] = (
        df.groupby("TeamName")["Points"].cumsum().shift(1).fillna(0)
    )

    # Points per race average
    df["points_per_race"] = df["driver_points_before_race"] / (
        df.groupby("DriverNumber").cumcount() + 1
    )
    df["points_per_race"] = df["points_per_race"].fillna(0)

    # Team average position per year
    df["team_avg_position"] = df.groupby(["Year", "TeamName"])["Position"].transform(
        "mean"
    )

    return df


def calculate_grid_advantage(df):
    """
    Calculate grid position advantage:
    - grid_advantage: Difference between average grid and current grid
    """
    df["grid_advantage"] = (
        df.groupby(["Year", "RoundNumber"])["GridPosition"].transform("mean")
        - df["GridPosition"]
    )
    return df


def engineer_all_features(input_file, output_file):
    """
    Main pipeline: Load data, create all features, save output

    Args:
        input_file: Path to cleaned race results CSV
        output_file: Path to save feature-engineered CSV
    """
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    df = load_raw_data(input_file)

    # Step 2: Create target variable
    print("\n[1/4] Creating target variable...")
    df = create_target_variable(df)

    # Step 3: Calculate recent form features
    print("[2/4] Calculating recent form metrics...")
    df = calculate_recent_form(df)

    # Step 4: Calculate cumulative points
    print("[3/4] Calculating cumulative points...")
    df = calculate_cumulative_points(df)

    # Step 5: Calculate grid advantage
    print("[4/4] Calculating grid advantage...")
    df = calculate_grid_advantage(df)

    # Step 6: Clean up and save
    print(f"\n Feature engineering complete!")
    print(f"Final shape: {df.shape}")
    print(f"Features created: {list(df.columns)}")

    # Save
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    return df


if __name__ == "__main__":
    engineer_all_features(
        input_file="data/f1_all_session_race_results.csv",
        output_file="data/f1_model_features.csv",
    )
