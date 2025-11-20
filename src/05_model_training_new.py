import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import warnings

warnings.filterwarnings("ignore")


def train_f1_models(
    input_file="data/f1_model_features.csv", model_output_dir="models/"
):
    """
    Train F1 prediction models and save them for future predictions.

    Args:
        input_file: Path to feature-engineered CSV
        model_output_dir: Directory to save trained models

    Returns:
        dict: Contains trained models and evaluation metrics
    """
    os.makedirs(model_output_dir, exist_ok=True)

    print("=" * 60)
    print("F1 RACE PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # Load the feature data
    df = pd.read_csv(input_file)

    print(f"\nDataset shape: {df.shape}")
    print(f"Years covered: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Total races: {df.groupby(['Year', 'RoundNumber']).ngroups}")

    # Check that all required features exist
    print("\n" + "=" * 60)
    print("VERIFYING FEATURES")
    print("=" * 60)

    required_features = [
        "IsTop3",
        "Position",
        "GridPosition",
        "driver_points_before_race",
        "team_points_before_race",
        "recent_avg_position",
        "recent_podiums",
        "position_consistency",
        "points_per_race",
        "team_avg_position",
        "grid_advantage",
    ]

    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        print(f"\n  Missing features: {missing_features}")
        print(
            "Please run feature engineering first: python src/04_feature_engineering.py"
        )
        return None

    print(" All required features present")
    print(f"Available features: {', '.join(required_features)}")

    # Prepare features for modeling
    print("\n" + "=" * 60)
    print("MODEL PREPARATION")
    print("=" * 60)

    # Encode team names
    le_team = LabelEncoder()
    df["team_encoded"] = le_team.fit_transform(df["TeamName"])

    # Select features
    feature_columns = [
        "GridPosition",
        "driver_points_before_race",
        "team_points_before_race",
        "recent_avg_position",
        "recent_podiums",
        "position_consistency",
        "points_per_race",
        "team_avg_position",
        "grid_advantage",
        "team_encoded",
    ]

    # Remove rows with missing values
    df_model = df[
        feature_columns
        + ["IsTop3", "DriverNumber", "FullName", "Year", "RoundNumber", "TeamName"]
    ].dropna()

    X = df_model[feature_columns]
    y = df_model["IsTop3"]

    print(f"\nTraining data shape: {X.shape}")
    print(f"Podium finishes (class 1): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Non-podium (class 0): {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

    # Split data - use 2023-2024 for training, 2025 for testing
    train_mask = df_model["Year"] < 2025
    test_mask = df_model["Year"] >= 2025

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"\nTrain set: {X_train.shape[0]} samples (years 2023-2024)")
    print(f"Test set: {X_test.shape[0]} samples (year 2025)")

    # Train Models
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    # Model 1: Random Forest
    print("\n1. Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)
    print("    Random Forest trained")

    # Model 2: Gradient Boosting
    print("2. Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
    )
    gb_model.fit(X_train, y_train)
    print("    Gradient Boosting trained")

    # Evaluate models
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Random Forest predictions
    rf_pred = rf_model.predict(X_test)
    print("\nRandom Forest Performance:")
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred, target_names=["Not Top 3", "Top 3"]))

    # Gradient Boosting predictions
    gb_pred = gb_model.predict(X_test)
    print("\nGradient Boosting Performance:")
    print(f"Accuracy: {accuracy_score(y_test, gb_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, gb_pred, target_names=["Not Top 3", "Top 3"]))

    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("=" * 60)

    feature_importance = pd.DataFrame(
        {"Feature": feature_columns, "Importance": rf_model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print("\n", feature_importance.to_string(index=False))

    # Save models
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)

    joblib.dump(rf_model, f"{model_output_dir}rf_model.pkl")
    joblib.dump(gb_model, f"{model_output_dir}gb_model.pkl")
    joblib.dump(le_team, f"{model_output_dir}le_team.pkl")
    joblib.dump(feature_columns, f"{model_output_dir}feature_columns.pkl")

    print(f"\n✓ Models saved to '{model_output_dir}':")
    print("  - rf_model.pkl")
    print("  - gb_model.pkl")
    print("  - le_team.pkl")
    print("  - feature_columns.pkl")

    return {
        "rf_model": rf_model,
        "gb_model": gb_model,
        "le_team": le_team,
        "feature_columns": feature_columns,
        "rf_accuracy": accuracy_score(y_test, rf_pred),
        "gb_accuracy": accuracy_score(y_test, gb_pred),
    }


if __name__ == "__main__":
    # Train models
    result = train_f1_models(
        input_file="data/f1_model_features.csv", model_output_dir="models/"
    )

    print("\n" + "=" * 60)
    print(" MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel Performance:")
    print(f"  Random Forest: {result['rf_accuracy']:.3f}")
    print(f"  Gradient Boosting: {result['gb_accuracy']:.3f}")
    print("\nNext step: Make predictions")
    print("  python src/06_prediction.py")
