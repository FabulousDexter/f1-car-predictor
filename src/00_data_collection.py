import fastf1
import pytz
import pandas as pd
import os
import json
from datetime import datetime

# Enable cache in a local folder
fastf1.Cache.enable_cache("./f1_cache")

CHECKPOINT_FILE = "data/checkpoint.json"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return {"successful_sessions": [], "failed_sessions": []}
                return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Warning: Corrupted checkpoint file. Starting fresh. Error: {e}")
            return {"successful_sessions": [], "failed_sessions": []}
        except Exception as e:
            print(f"Warning: Could not read checkpoint file. Error: {e} ")
            return {"successful_sessions": [], "failed_sessions": []}
    return {"successful_sessions": [], "failed_sessions": []}


def save_checkpoint(checkpoint_data):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def is_session_fetched(year, round_number, session_type, checkpoint_data):
    session_id = f"{year}_{round_number}_{session_type}"
    return session_id in checkpoint_data.get("successful_sessions", [])


def mark_session_fetched(
    year, round_number, session_type, checkpoint_data, success=True
):
    session_id = f"{year}_{round_number}_{session_type}"
    if success:
        if session_id not in checkpoint_data.get("successful_sessions", []):
            checkpoint_data["successful_sessions"].append(session_id)
        if session_id in checkpoint_data.get("failed_sessions", []):
            checkpoint_data["failed_sessions"].remove(session_id)
    else:
        if session_id not in checkpoint_data.get("failed_sessions", []):
            checkpoint_data["failed_sessions"].append(session_id)


def load_session_with_retry(
    year, round_number, session_type, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY
):
    import time

    for attempt in range(max_retries):
        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load()
            return session, None
        except Exception as e:
            error_msg = str(e)

            if (
                "Failed to load schedule from" in error_msg
                or "Failed to load" in error_msg
            ):
                if attempt < max_retries - 1:
                    print(
                        f"   Retry {attempt + 1}/{max_retries} after {RETRY_DELAY} seconds..."
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    return (
                        None,
                        f"API failure after {max_retries} attempts: {error_msg}",
                    )
            else:
                return None, error_msg
    return None, "Max retries exceeded"


def get_all_event_schedule(start_year: int, end_year: int = None):
    if end_year is None:
        end_year = datetime.now().year

    event_schedule_list = []

    for year in range(start_year, end_year + 1):
        current_event_schedule = fastf1.get_event_schedule(year, include_testing=False)
        if year == datetime.now().year:
            current_event_schedule["Session5DateUtc"] = pd.to_datetime(
                current_event_schedule["Session5DateUtc"]
            ).dt.tz_localize("UTC")

            now_utc = datetime.now(pytz.utc)
            completed_current_year = current_event_schedule[
                current_event_schedule["Session5DateUtc"] < now_utc
            ].copy()

            event_schedule_list.append(completed_current_year)
        else:
            event_schedule_list.append(current_event_schedule)

    all_events = pd.concat(event_schedule_list)
    all_events.reset_index(drop=True, inplace=True)
    return all_events


def save_dateframe_to_csv(df: pd.DataFrame, filename):
    # Save input to csv file
    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Save {len(df)} rows to {filename}")


def get_all_sessions(
    start_year: int = 2023, end_year: int = None, retry_failed: bool = False
):
    # Get all sessions
    checkpoint_data = load_checkpoint()
    successful_session, failed_session = [], []

    all_events = get_all_event_schedule(start_year, end_year)

    results_filename = f"data/f1_all_session_race_results_{start_year}-{end_year}.csv"
    laps_filename = f"data/f1_all_session_race_laps_{start_year}-{end_year}.csv"

    all_results = []
    all_laps = []

    if os.path.exists(results_filename):
        existing_results = pd.read_csv(results_filename)
        all_results.append(existing_results)
        print(
            f"Loaded {len(existing_results)} existing results from {results_filename}"
        )

    if os.path.exists(laps_filename):
        existing_laps = pd.read_csv(laps_filename)
        all_laps.append(existing_laps)
        print(f"Loaded {len(existing_laps)} existing laps from {laps_filename}")

    session_types = ["FP1", "FP2", "FP3", "Q", "R"]
    results_columns = [
        "DriverNumber",
        "FullName",
        "Position",
        "TeamName",
        "GridPosition",
        "Q1",
        "Q2",
        "Q3",
        "Time",
        "Status",
        "Points",
    ]
    lap_columns = ["Driver", "LapNumber", "LapTime", "Compound", "TyreLife"]

    for index, event_row in all_events.iterrows():
        year = event_row["EventDate"].year
        round_number = event_row["RoundNumber"]
        print("=" * 50)
        print(f"[{index+1}/{len(all_events)}] Loading {event_row['EventName']}...")
        print("=" * 50)

        for session_type in session_types:
            session_id = f"{year}_{round_number}_{session_type}"
            if is_session_fetched(year, round_number, session_type, checkpoint_data):
                print(
                    f"  Skipping {session_type} for {event_row['EventName']} (already fetched)"
                )
                continue

            if not retry_failed and session_id in checkpoint_data.get(
                "failed_sessions", []
            ):
                print(
                    f"  Skipping {session_type} for {event_row['EventName']} (previously failed). Use retry_failed=True to retry."
                )
                continue

            session, error = load_session_with_retry(year, round_number, session_type)
            if session is None:
                print(
                    f"  Could not load {session_type} for {event_row['EventName']}: {error}"
                )
                failed_session.append(
                    (f"{event_row['EventName']} - {session_type}", year)
                )
                mark_session_fetched(
                    year, round_number, session_type, checkpoint_data, success=False
                )
                save_checkpoint(checkpoint_data)
                continue

            try:

                available_results_cols = [
                    col for col in results_columns if col in session.results.columns
                ]
                results = session.results[available_results_cols].copy()
                # Add metadata columns to results
                results["Year"] = year
                results["EventName"] = event_row["EventName"]
                results["RoundNumber"] = round_number
                results["SessionType"] = session_type
                all_results.append(results)

                if not session.laps.empty:
                    available_lap_cols = [
                        col for col in lap_columns if col in session.laps.columns
                    ]
                    laps = session.laps[available_lap_cols].copy()
                    # Add metadata columns to laps
                    laps["Year"] = year
                    laps["EventName"] = event_row["EventName"]
                    laps["RoundNumber"] = round_number
                    laps["SessionType"] = session_type
                    all_laps.append(laps)

                print(
                    f"Successfully loaded {session_type} for {event_row['EventName']}"
                )
                successful_session.append(
                    (f"{event_row['EventName']} - {session_type}", year)
                )
                mark_session_fetched(
                    year, round_number, session_type, checkpoint_data, success=True
                )

                save_checkpoint(checkpoint_data)

            except Exception as e:
                print(
                    f"  Could not load {session_type} for {event_row['EventName']}: {e}"
                )
                failed_session.append(
                    (f"{event_row['EventName']} - {session_type}", year)
                )
                mark_session_fetched(
                    year, round_number, session_type, checkpoint_data, success=False
                )
                save_checkpoint(checkpoint_data)
                continue

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df = results_df.drop_duplicates(
            subset=["Year", "RoundNumber", "SessionType", "DriverNumber"]
        )

        laps_df = pd.concat(all_laps, ignore_index=True)
        laps_df = laps_df.drop_duplicates(
            subset=["Year", "RoundNumber", "SessionType", "Driver", "LapNumber"]
        )

        save_dateframe_to_csv(
            results_df, f"data/f1_all_session_race_results.csv"
        )
        save_dateframe_to_csv(
            laps_df, f"data/f1_all_session_race_laps.csv"
        )
    else:
        print("No data to save!")

    print("\nSummary:")
    print(f"Successful sessions: {len(successful_session)}")
    print(f"Failed sessions: {len(failed_session)}")

    if failed_session:
        print("Failed sessions:")
        for session_name, year in failed_session:
            print(f" - {session_name} ({year})")
        print(
            f"\nTo retry failed sessions, set retry_failed=True in get_all_sessions()."
        )
        print(
            f"Example: get_all_sessions(start_year={start_year}, end_year={end_year}, retry_failed=True)"
        )


if __name__ == "__main__":
    # get_all_sessions()
    get_all_sessions(start_year=2023, end_year=2025)
