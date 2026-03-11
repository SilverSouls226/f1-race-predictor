import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm

YEAR = 2018
fastf1.Cache.enable_cache("cache")

def to_seconds(td):
    if pd.isna(td):
        return np.nan
    return td.total_seconds()
all_rows = []
schedule = fastf1.get_event_schedule(YEAR)

for _, event in tqdm(schedule.iterrows(), desc=f"Season {YEAR}"):

    if event["EventFormat"] == "testing":
        continue

    race_name = event["EventName"]
    event_date = event["EventDate"]
    circuit_name = event["Location"]
    country = event["Country"]

    try:
        race = fastf1.get_session(YEAR, race_name, "R")
        race.load(laps=True, weather=True, telemetry=False, messages=False)

        quali = fastf1.get_session(YEAR, race_name, "Q")
        quali.load(laps=False, weather=False)

    except Exception as e:
        print(f"Skipping {race_name} due to error: {e}")
        continue

    race_results = race.results
    quali_results = quali.results
    laps = race.laps.copy()
    weather = race.weather_data

    quali_map = {
        row["Abbreviation"]: row
        for _, row in quali_results.iterrows()
    }

    for _, driver_row in race_results.iterrows():

        drv = driver_row["Abbreviation"]
        team = driver_row["TeamName"]

        driver_laps = laps.pick_drivers(drv)

        if len(driver_laps) == 0:
            continue

        qrow = quali_map.get(drv)

        qualifying_position = np.nan
        qualifying_time = np.nan

        if qrow is not None:
            qualifying_position = qrow["Position"]

            times = [
                to_seconds(qrow["Q1"]),
                to_seconds(qrow["Q2"]),
                to_seconds(qrow["Q3"])
            ]
            times = [t for t in times if not np.isnan(t)]
            if len(times) > 0:
                qualifying_time = min(times)

        for _, lap in driver_laps.iterrows():

            weather_slice = weather.iloc[
                (weather["Time"] - lap["Time"]).abs().argsort()[:1]
            ]

            row = {
                "season": YEAR,
                "race_name": race_name,
                "event_date": event_date,
                "circuit_name": circuit_name,
                "country": country,

                "driver": drv,
                "team": team,

                "grid_position": driver_row["GridPosition"],
                "qualifying_position": qualifying_position,
                "qualifying_time": qualifying_time,

                "finish_position": driver_row["Position"],
                "points_scored": driver_row["Points"],
                "championship_points": driver_row["Points"],
                "status": driver_row["Status"],

                "lap_number": lap["LapNumber"],
                "lap_time": to_seconds(lap["LapTime"]),
                "sector1_time": to_seconds(lap["Sector1Time"]),
                "sector2_time": to_seconds(lap["Sector2Time"]),
                "sector3_time": to_seconds(lap["Sector3Time"]),

                "tyre_compound": lap["Compound"],
                "tyre_life": lap["TyreLife"],
                "is_new_tyre": lap["FreshTyre"],

                "pit_in_time": to_seconds(lap["PitInTime"]),
                "pit_out_time": to_seconds(lap["PitOutTime"]),
                "pit_duration": (
                    to_seconds(lap["PitOutTime"] - lap["PitInTime"])
                    if pd.notna(lap["PitInTime"]) and pd.notna(lap["PitOutTime"])
                    else np.nan
                ),

                "air_temperature": weather_slice["AirTemp"].values[0],
                "track_temperature": weather_slice["TrackTemp"].values[0],
                "humidity": weather_slice["Humidity"].values[0],
                "pressure": weather_slice["Pressure"].values[0],
                "rainfall": weather_slice["Rainfall"].values[0],
                "wind_speed": weather_slice["WindSpeed"].values[0],
                "wind_direction": weather_slice["WindDirection"].values[0],

                "track_status": lap["TrackStatus"]
            }

            all_rows.append(row)

df = pd.DataFrame(all_rows)
outfile = f"season_{YEAR}.csv"
df.to_csv(outfile, index=False)
print(f"\nSaved {outfile} with {len(df):,} rows")