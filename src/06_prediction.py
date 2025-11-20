import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime
import fastf1

warnings.filterwarnings("ignore")

# Enable FastF1 cache
fastf1.Cache.enable_cache("./f1_cache")


def load_models(model_dir="models/"):
    """
    Load trained models and metadata.

    Args:
        model_dir: Directory containing saved models

    Returns:
        dict: Contains models and feature information
    """
    print("=" * 60)
    print("LOADING TRAINED MODELS")
    print("=" * 60)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    rf_model = joblib.load(f"{model_dir}rf_model.pkl")
    gb_model = joblib.load(f"{model_dir}gb_model.pkl")
    le_team = joblib.load(f"{model_dir}le_team.pkl")
    feature_columns = joblib.load(f"{model_dir}feature_columns.pkl")

    print("✓ Models loaded successfully:")
    print("  - Random Forest")
    print("  - Gradient Boosting")
    print("  - Team Label Encoder")
    print(f"  - {len(feature_columns)} features")

    return {
        "rf_model": rf_model,
        "gb_model": gb_model,
        "le_team": le_team,
        "feature_columns": feature_columns,
    }


def prepare_driver_features(df, le_team, feature_columns):
    """
    Prepare features for prediction.
    All features should already be calculated by feature engineering pipeline.
    This function only encodes team names.

    Args:
        df: DataFrame with driver data (features already calculated)
        le_team: Label encoder for teams
        feature_columns: List of required features

    Returns:
        DataFrame: Prepared features ready for prediction
    """
    # Verify all required features exist
    missing_features = [
        f for f in feature_columns if f not in df.columns and f != "team_encoded"
    ]
    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}. Run feature engineering first."
        )

    # Only encode team names (all other features should already exist)
    df["team_encoded"] = le_team.transform(df["TeamName"])

    return df


def get_qualifying_grid(year, round_number, max_retries=3):
    """
    Fetch Saturday's qualifying results with finalized grid positions.
    Run this on Saturday evening after Q3 and penalties are applied.

    Args:
        year: Race year
        round_number: Race round number
        max_retries: Number of retry attempts for API calls

    Returns:
        DataFrame: Grid positions for each driver with columns:
                   [DriverNumber, FullName, TeamName, GridPosition]
    """
    import time

    print("\n" + "=" * 60)
    print("FETCHING QUALIFYING GRID (SATURDAY)")
    print("=" * 60)

    for attempt in range(max_retries):
        try:
            # Load qualifying session
            qualifying = fastf1.get_session(year, round_number, "Q")
            qualifying.load()

            # Get results with grid positions (includes penalties)
            results = qualifying.results[
                ["DriverNumber", "FullName", "TeamName", "Position"]
            ].copy()

            # Rename Position to GridPosition for clarity
            results.rename(columns={"Position": "GridPosition"}, inplace=True)

            print(f"✓ Successfully fetched qualifying grid for Round {round_number}")
            print(f"  Found {len(results)} drivers")
            print(
                f"  Pole Position: {results.iloc[0]['FullName']} ({results.iloc[0]['TeamName']})"
            )

            return results

        except Exception as e:
            error_msg = str(e)
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {error_msg}")

            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # Progressive backoff
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(
                    "\n❌ Failed to fetch qualifying grid. Using historical averages instead."
                )
                return None

    return None


def get_next_race(features_file="data/f1_model_features.csv"):
    """
    Automatically detect the next race to predict.

    Args:
        features_file: Path to features CSV with all race data

    Returns:
        tuple: (year, round_number, race_name) of next race to predict
    """
    df = pd.read_csv(features_file)

    # Get the latest race in the data
    latest_year = df["Year"].max()
    latest_round = df[df["Year"] == latest_year]["RoundNumber"].max()

    # Next race is latest_round + 1
    next_round = latest_round + 1

    # Check if we have any races with this round number (to get race name)
    race_info = df[
        (df["Year"] == latest_year) & (df["RoundNumber"] == latest_round)
    ].iloc[0]

    print("\n" + "=" * 60)
    print("DETECTING NEXT RACE")
    print("=" * 60)
    print(f"Latest completed race: {race_info['EventName']} (Round {latest_round})")
    print(f"Next race to predict: Round {next_round} of {latest_year}")

    return latest_year, next_round, f"Round {next_round}"


def predict_race(
    year,
    round_number,
    features_file="data/f1_model_features.csv",
    model_dir="models/",
    output_file=None,
    use_qualifying_grid=False,
):
    """
    Predict race results for a specific race.

    Args:
        year: Year of the race
        round_number: Round number of the race
        features_file: Path to features CSV
        model_dir: Directory containing trained models
        output_file: Optional path to save predictions
        use_qualifying_grid: If True, fetch actual grid from Saturday qualifying

    Returns:
        DataFrame: Predictions with probabilities
    """
    # Load models
    models = load_models(model_dir)
    rf_model = models["rf_model"]
    gb_model = models["gb_model"]
    le_team = models["le_team"]
    feature_columns = models["feature_columns"]

    # Load feature data
    print("\n" + "=" * 60)
    print(f"PREPARING PREDICTION FOR {year} ROUND {round_number}")
    print("=" * 60)

    df = pd.read_csv(features_file)

    # Prepare features
    df = prepare_driver_features(df, le_team, feature_columns)

    # Get the most recent race data before the target race
    available_data = df[(df["Year"] == year) & (df["RoundNumber"] < round_number)]

    if len(available_data) == 0:
        print(f"\n  No data available for {year} before round {round_number}")
        print("Using latest available data from previous races...")
        latest_race = df[df["Year"] == year]["RoundNumber"].max()
    else:
        latest_race = available_data["RoundNumber"].max()

    print(f"\nUsing data up to Round {latest_race} of {year} season")

    # Optionally fetch actual qualifying grid from Saturday
    if use_qualifying_grid:
        print("\n📡 Fetching actual qualifying grid from Saturday...")
        qualifying_grid = get_qualifying_grid(year, round_number)

        if qualifying_grid is not None:
            print("✓ Using actual grid positions from qualifying session")
            # Will merge actual grid positions with driver features below
        else:
            print("⚠ Using predicted grid positions instead")
            qualifying_grid = None
    else:
        qualifying_grid = None

    # Get latest driver statistics (aggregate by driver to avoid duplicates)
    latest_data = df[(df["Year"] == year) & (df["RoundNumber"] == latest_race)].copy()

    # Aggregate numeric features by taking the mean for each driver
    agg_dict = {col: "mean" for col in feature_columns}
    agg_dict.update({"FullName": "first", "TeamName": "first"})

    latest_stats = latest_data.groupby("DriverNumber").agg(agg_dict).reset_index()

    print(f"Active drivers: {len(latest_stats)}")

    # Update with actual qualifying grid if available
    if qualifying_grid is not None:
        print("\n🔄 Updating driver features with actual grid positions...")
        # Merge actual grid positions into driver stats
        latest_stats = latest_stats.merge(
            qualifying_grid[["DriverNumber", "GridPosition"]],
            on="DriverNumber",
            how="left",
            suffixes=("_old", ""),
        )
        # Replace old GridPosition with actual one
        if "GridPosition_old" in latest_stats.columns:
            latest_stats["GridPosition"] = latest_stats["GridPosition"].fillna(
                latest_stats["GridPosition_old"]
            )
            latest_stats.drop(columns=["GridPosition_old"], inplace=True)

        # Recalculate grid_advantage with actual grid
        if (
            "GridPosition" in latest_stats.columns
            and "grid_advantage" in feature_columns
        ):
            avg_grid = latest_stats["GridPosition"].mean()
            latest_stats["grid_advantage"] = avg_grid - latest_stats["GridPosition"]
            print(f"✓ Grid advantage recalculated (avg grid: {avg_grid:.1f})")

    # Prepare prediction data
    X_pred = latest_stats[feature_columns]

    # Handle NaN values
    print(f"\nChecking for missing values...")
    nan_counts = X_pred.isna().sum()
    if nan_counts.any():
        print("  Found NaN values, filling with defaults:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"  - {col}: {nan_counts[col]} NaN values")
        X_pred = X_pred.fillna(0)
        print("✓ NaN values filled with 0")
    else:
        print("✓ No missing values found")

    # Make predictions
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    rf_proba = rf_model.predict_proba(X_pred)[:, 1]
    gb_proba = gb_model.predict_proba(X_pred)[:, 1]
    ensemble_proba = (rf_proba + gb_proba) / 2

    # Create prediction dataframe
    predictions = latest_stats[["DriverNumber", "FullName", "TeamName"]].copy()
    predictions["Podium_Probability_RF"] = rf_proba
    predictions["Podium_Probability_GB"] = gb_proba
    predictions["Podium_Probability_Ensemble"] = ensemble_proba

    # Sort by ensemble probability
    predictions = predictions.sort_values(
        "Podium_Probability_Ensemble", ascending=False
    )
    predictions = predictions.reset_index(drop=True)

    # Display top 3
    print("\n" + "=" * 60)
    print(f"🏆 TOP 3 PREDICTED WINNERS")
    print("=" * 60)

    top_3 = predictions.head(3)
    for idx, row in top_3.iterrows():
        print(f"\n{idx+1}. {row['FullName']} ({row['TeamName']})")
        print(f"   Podium Probability: {row['Podium_Probability_Ensemble']*100:.1f}%")
        print(
            f"   RF: {row['Podium_Probability_RF']*100:.1f}% | GB: {row['Podium_Probability_GB']*100:.1f}%"
        )

    # Display top 10
    print("\n" + "=" * 60)
    print("TOP 10 PREDICTIONS")
    print("=" * 60)
    print(
        "\n",
        predictions.head(10)[
            ["FullName", "TeamName", "Podium_Probability_Ensemble"]
        ].to_string(index=False),
    )

    # Model agreement
    print("\n" + "=" * 60)
    print("MODEL CONFIDENCE METRICS")
    print("=" * 60)

    print(
        f"\nTop 3 average confidence: {top_3['Podium_Probability_Ensemble'].mean()*100:.1f}%"
    )
    print("Model agreement (RF vs GB top 3):")
    rf_top3 = set(predictions.nlargest(3, "Podium_Probability_RF")["FullName"])
    gb_top3 = set(predictions.nlargest(3, "Podium_Probability_GB")["FullName"])
    agreement = rf_top3.intersection(gb_top3)
    print(f"  Both models agree on: {', '.join(agreement) if agreement else 'None'}")
    print(f"  Agreement rate: {len(agreement)}/3 drivers")

    # Save predictions
    if output_file:
        predictions.to_csv(output_file, index=False)
        print(f"\n✅ Predictions saved to '{output_file}'")

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 Race Prediction")
    parser.add_argument(
        "--auto", action="store_true", help="Automatically detect and predict next race"
    )
    parser.add_argument(
        "--year", type=int, help="Year of race to predict (if not using --auto)"
    )
    parser.add_argument(
        "--round",
        type=int,
        help="Round number of race to predict (if not using --auto)",
    )
    parser.add_argument(
        "--use-qualifying",
        action="store_true",
        help="Fetch actual grid positions from Saturday qualifying (run after Q3)",
    )

    args = parser.parse_args()

    # Determine which race to predict
    if args.auto:
        # Auto-detect next race
        year, round_num, race_name = get_next_race("data/f1_model_features.csv")
        output_file = f"data/predictions_round_{round_num}_{year}.csv"
    elif args.year and args.round:
        # User specified race
        year = args.year
        round_num = args.round
        output_file = f"data/predictions_round_{round_num}_{year}.csv"
    else:
        # Default: Auto-detect
        print("No race specified, auto-detecting next race...")
        year, round_num, race_name = get_next_race("data/f1_model_features.csv")
        output_file = f"data/predictions_round_{round_num}_{year}.csv"

    # Run prediction
    predictions = predict_race(
        year=year,
        round_number=round_num,
        features_file="data/f1_model_features.csv",
        model_dir="models/",
        output_file=output_file,
        use_qualifying_grid=args.use_qualifying,
    )

    print("\n" + "=" * 60)
    print("✅ PREDICTION COMPLETE!")
    print("=" * 60)
