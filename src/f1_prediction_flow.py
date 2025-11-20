"""
F1 Prediction Pipeline with Prefect Orchestration

This flow orchestrates the entire F1 prediction pipeline:
1. Data collection
2. Data validation
3. Data cleaning
4. Feature engineering
5. Model training
6. Prediction generation
"""

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import pipeline functions directly
import importlib.util


def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


src_dir = Path(__file__).parent
dc_module = load_module("data_collection", src_dir / "01_data_collection.py")
dv_module = load_module("data_validation", src_dir / "02_data_validation.py")
dcl_module = load_module("data_cleaning", src_dir / "03_data_cleaning.py")
fe_module = load_module("feature_engineering", src_dir / "04_feature_engineering.py")
mt_module = load_module("model_training", src_dir / "05_model_training_new.py")
pred_module = load_module("prediction", src_dir / "06_prediction.py")


@task(name="Check Data Collection Status", retries=2, retry_delay_seconds=5)
def check_data_status():
    """Check if data collection is up to date"""
    checkpoint = dc_module.load_checkpoint()
    failed_sessions = [
        s
        for s in checkpoint.get("failed_sessions", [])
        if "Practice" not in s  # Ignore practice session failures
    ]

    total_sessions = len(checkpoint.get("successful_sessions", [])) + len(
        failed_sessions
    )
    success_rate = len(checkpoint.get("successful_sessions", [])) / max(
        total_sessions, 1
    )

    print(f"Data collection status:")
    print(f"  Successful sessions: {len(checkpoint.get('successful_sessions', []))}")
    print(f"  Failed sessions: {len(failed_sessions)}")
    print(f"  Success rate: {success_rate*100:.1f}%")

    return {
        "needs_collection": len(failed_sessions) > 0 or success_rate < 0.95,
        "checkpoint": checkpoint,
    }


@task(name="Collect F1 Data", retries=3, retry_delay_seconds=10)
def collect_data_task(retry_failed: bool = False):
    """Collect F1 race data with retry logic"""
    print("\n" + "=" * 60)
    print("TASK: Data Collection")
    print("=" * 60)

    dc_module.get_all_sessions(
        start_year=2023, end_year=2025, retry_failed=retry_failed
    )

    return {"status": "success", "step": "collection"}


@task(name="Validate Data", retries=2)
def validate_data_task():
    """Validate collected data quality"""
    print("\n" + "=" * 60)
    print("TASK: Data Validation")
    print("=" * 60)

    # Validate race results
    race_results_valid = dv_module.validate_race_results(
        "data/f1_all_session_race_results.csv"
    )

    # Validate lap times
    lap_times_valid = dv_module.validate_lap_times("data/f1_all_session_race_laps.csv")

    validation_results = {
        "race_results": race_results_valid,
        "lap_times": lap_times_valid,
    }

    return {"status": "success", "step": "validation", "results": validation_results}


@task(name="Clean Race Results")
def clean_race_results_task():
    """Clean race results data"""
    print("\n" + "=" * 60)
    print("TASK: Clean Race Results")
    print("=" * 60)

    dcl_module.clean_race_results(
        input_file="data/f1_all_session_race_results.csv",
        output_file="data/f1_all_session_race_results_clean.csv",
    )

    return {"status": "success", "step": "clean_results"}


@task(name="Clean Lap Times")
def clean_lap_times_task():
    """Clean lap times data"""
    print("\n" + "=" * 60)
    print("TASK: Clean Lap Times")
    print("=" * 60)

    dcl_module.clean_lap_times(
        input_file="data/f1_all_session_race_laps.csv",
        output_file="data/f1_all_session_race_laps_clean.csv",
    )

    return {"status": "success", "step": "clean_laps"}


@task(name="Engineer Features", retries=2)
def engineer_features_task():
    """Create model features from cleaned data"""
    print("\n" + "=" * 60)
    print("TASK: Feature Engineering")
    print("=" * 60)

    fe_module.engineer_all_features(
        input_file="data/f1_all_session_race_results.csv",
        output_file="data/f1_model_features.csv",
    )

    return {"status": "success", "step": "features"}


@task(name="Train Models", retries=2)
def train_models_task():
    """Train prediction models"""
    print("\n" + "=" * 60)
    print("TASK: Model Training")
    print("=" * 60)

    result = mt_module.train_f1_models(
        input_file="data/f1_model_features.csv", model_output_dir="models/"
    )

    return {
        "status": "success",
        "step": "training",
        "rf_accuracy": result["rf_accuracy"],
        "gb_accuracy": result["gb_accuracy"],
    }


@task(name="Generate Predictions")
def predict_race_task(year: int = None, round_number: int = None, auto: bool = True):
    """Generate race predictions - auto-detects next race by default"""
    print("\n" + "=" * 60)
    print("TASK: Prediction Generation")
    print("=" * 60)

    # Auto-detect next race if not specified
    if auto and (year is None or round_number is None):
        year, round_number, race_name = pred_module.get_next_race(
            "data/f1_model_features.csv"
        )
        print(f"\n🔍 Auto-detected next race: {race_name}")

    predictions = pred_module.predict_race(
        year=year,
        round_number=round_number,
        features_file="data/f1_model_features.csv",
        model_dir="models/",
        output_file=f"data/predictions_round_{round_number}_{year}.csv",
    )

    return {
        "status": "success",
        "step": "prediction",
        "year": year,
        "round": round_number,
        "top_3": predictions.head(3)[
            ["FullName", "TeamName", "Podium_Probability_Ensemble"]
        ].to_dict(),
    }


@flow(
    name="F1 Full Pipeline",
    description="Complete F1 prediction pipeline from data collection to prediction",
    task_runner=ConcurrentTaskRunner(),
)
def f1_full_pipeline(
    collect_new_data: bool = False,
    retry_failed: bool = False,
    auto_detect_race: bool = True,
    target_year: int = None,
    target_round: int = None,
):
    """
    Run the complete F1 prediction pipeline.

    Args:
        collect_new_data: Whether to collect new data
        retry_failed: Whether to retry failed data collection sessions
        auto_detect_race: Automatically detect next race to predict
        target_year: Year for prediction (only if auto_detect_race=False)
        target_round: Round number for prediction (only if auto_detect_race=False)
    """
    print("\n" + "=" * 60)
    print("F1 PREDICTION PIPELINE - PREFECT ORCHESTRATION")
    print("=" * 60)

    # Step 1: Check data status
    data_status = check_data_status()

    # Step 2: Collect data if needed
    if collect_new_data or data_status["needs_collection"]:
        collect_data_task(retry_failed=retry_failed)
    else:
        print("\n✓ Skipping data collection (data is up to date)")

    # Step 3: Validate data
    validation_result = validate_data_task()

    # Step 4: Clean data (can run in parallel)
    clean_results = clean_race_results_task()
    clean_laps = clean_lap_times_task()

    # Wait for both cleaning tasks to complete
    # (Prefect handles this automatically with task dependencies)

    # Step 5: Feature engineering (depends on cleaning)
    features_result = engineer_features_task()

    # Step 6: Train models (depends on features)
    training_result = train_models_task()

    # Step 7: Generate predictions (depends on training)
    prediction_result = predict_race_task(
        year=target_year, round_number=target_round, auto=auto_detect_race
    )

    # Summary
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nModel Performance:")
    print(f"  Random Forest: {training_result['rf_accuracy']:.3f}")
    print(f"  Gradient Boosting: {training_result['gb_accuracy']:.3f}")
    print(
        f"\nPrediction: Round {prediction_result['round']} of {prediction_result['year']}"
    )
    print(
        f"Output saved to: data/predictions_round_{prediction_result['round']}_{prediction_result['year']}.csv"
    )

    return {
        "validation": validation_result,
        "training": training_result,
        "prediction": prediction_result,
    }


@flow(name="F1 Training Only", description="Train models without data collection")
def f1_training_pipeline():
    """
    Run only the training portion of the pipeline.
    Assumes data is already collected and cleaned.
    """
    print("\n" + "=" * 60)
    print("F1 TRAINING PIPELINE")
    print("=" * 60)

    # Feature engineering
    features_result = engineer_features_task()

    # Model training
    training_result = train_models_task()

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel Performance:")
    print(f"  Random Forest: {training_result['rf_accuracy']:.3f}")
    print(f"  Gradient Boosting: {training_result['gb_accuracy']:.3f}")

    return training_result


@flow(
    name="F1 Prediction Only", description="Generate predictions using existing models"
)
def f1_prediction_pipeline(
    auto_detect: bool = True, year: int = None, round_number: int = None
):
    """
    Run only prediction generation.
    Assumes models are already trained.

    Args:
        auto_detect: Automatically detect next race (default: True)
        year: Year for prediction (only if auto_detect=False)
        round_number: Round number for prediction (only if auto_detect=False)
    """
    print("\n" + "=" * 60)
    print("F1 PREDICTION PIPELINE")
    print("=" * 60)

    prediction_result = predict_race_task(
        year=year, round_number=round_number, auto=auto_detect
    )

    print("\n" + "=" * 60)
    print("✅ PREDICTION COMPLETE!")
    print("=" * 60)
    print(
        f"\nPredicted: Round {prediction_result['round']} of {prediction_result['year']}"
    )

    return prediction_result

    return prediction_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 Prediction Pipeline with Prefect")
    parser.add_argument(
        "--mode",
        choices=["full", "training", "prediction"],
        default="full",
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--collect-data", action="store_true", help="Force data collection"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed data collection sessions",
    )
    parser.add_argument(
        "--year", type=int, default=2025, help="Target year for prediction"
    )
    parser.add_argument(
        "--round", type=int, default=22, help="Target round number for prediction"
    )

    args = parser.parse_args()

    if args.mode == "full":
        f1_full_pipeline(
            collect_new_data=args.collect_data,
            retry_failed=args.retry_failed,
            target_year=args.year,
            target_round=args.round,
        )
    elif args.mode == "training":
        f1_training_pipeline()
    elif args.mode == "prediction":
        f1_prediction_pipeline(year=args.year, round_number=args.round)
