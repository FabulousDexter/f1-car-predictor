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


def get_all_sessions(start_year: int = 2023):
    # Get all sessions
    successful_load, failed_load = 0, 0
    failed_session = []

    all_events = get_all_event_schedule(start_year)
    all_results = []
    all_laps = []

    for index, event_row in all_events.iterrows():
        year = event_row["EventDate"].year
        round_number = event_row["RoundNumber"]
        print("=" * 50)
        print(f"[{index+1}/{len(all_events)}] Loading {event_row['EventName']}...")
        print("=" * 50)

        try:
            session = fastf1.get_session(year, round_number, "R")
            session.load()

            results = session.results[
                ["DriverNumber", "FullName", "Position", "TeamName"]
            ].copy()
            # Add metadata columns to results
            results["Year"] = year
            results["EventName"] = event_row["EventName"]
            results["RoundNumber"] = round_number
            all_results.append(results)

            laps = session.laps[
                ["Driver", "LapNumber", "LapTime", "Compound", "TyreLife"]
            ].copy()
            # Add metadata columns to laps
            laps["Year"] = year
            laps["EventName"] = event_row["EventName"]
            laps["RoundNumber"] = round_number
            all_laps.append(laps)

            print(session)
            print(f"Loaded {event_row['EventName']}")
            successful_load += 1
        except Exception as e:
            print(f"Failed {event_row['EventName']}: {e}")
            failed_session.append((event_row["EventName"], year))
            failed_load += 1
            continue

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        laps_df = pd.concat(all_laps, ignore_index=True)

        save_dateframe_to_csv(results_df, "data/race_results.csv")
        save_dateframe_to_csv(laps_df, "data/race_laps.csv")
    else:
        print("No data to save!")

    print(f"Successful load: {successful_load}")
    print(f"Failed load: {failed_load}")


if __name__ == "__main__":
    get_all_sessions()
