import fastf1
import pytz
import pandas as pd
import os
from datetime import datetime

# Enable cache in a local folder
fastf1.Cache.enable_cache("./f1_cache")


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


def get_all_sessions(start_year: int = 2023, end_year: int = None):
    # Get all sessions
    successful_session, failed_session = [], []

    all_events = get_all_event_schedule(start_year, end_year)
    all_results = []
    all_laps = []
    session_types = ["FP1", "FP2", "FP3", "Q", "R"]
    results_columns = ["DriverNumber", "FullName", "Position", "TeamName", "GridPosition", "Q1", "Q2", "Q3", "Time", "Status", "Points"]
    lap_columns = ["Driver", "LapNumber", "LapTime", "Compound", "TyreLife"]

    for index, event_row in all_events.iterrows():
        year = event_row["EventDate"].year
        round_number = event_row["RoundNumber"]
        print("=" * 50)
        print(f"[{index+1}/{len(all_events)}] Loading {event_row['EventName']}...")
        print("=" * 50)

        for session_type in session_types:
            try:
                session = fastf1.get_session(year, round_number, session_type)
                session.load()

                available_results_cols = [col for col in results_columns if col in session.results.columns]
                results = session.results[available_results_cols].copy()
                # Add metadata columns to results
                results["Year"] = year
                results["EventName"] = event_row["EventName"]
                results["RoundNumber"] = round_number
                results["SessionType"] = session_type
                all_results.append(results)
                
                if not session.laps.empty:
                    available_lap_cols = [col for col in lap_columns if col in session.laps.columns]
                    laps = session.laps[available_lap_cols].copy()
                    # Add metadata columns to laps
                    laps["Year"] = year
                    laps["EventName"] = event_row["EventName"]
                    laps["RoundNumber"] = round_number
                    laps["SessionType"] = session_type
                    all_laps.append(laps)
                
                print(f"Successfully loaded {session_type} for {event_row['EventName']}")
                successful_session.append((f"{event_row['EventName']} - {session_type}", year))

            except Exception as e:
                print(f"  Could not load {session_type} for {event_row['EventName']}: {e}")
                failed_session.append((f"{event_row['EventName']} - {session_type}", year))
                continue                
        
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        laps_df = pd.concat(all_laps, ignore_index=True)

        save_dateframe_to_csv(results_df, f"data/f1_all_session_race_results_{start_year}-{end_year}.csv")
        save_dateframe_to_csv(laps_df, f"data/f1_all_session_race_laps_{start_year}-{end_year}.csv")
    else:
        print("No data to save!")

    print("\nSummary:")
    print(f"Successful sessions: {len(successful_session)}")
    print(f"Failed sessions: {len(failed_session)}")
    
    if failed_session:
        print("Failed sessions:")
        for session_name, year in failed_session:
            print(f" - {session_name} ({year})")


if __name__ == "__main__":
    #get_all_sessions()
    get_all_sessions(start_year=2023, end_year=2025)    
