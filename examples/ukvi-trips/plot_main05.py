"""
UKVI Travel History Calculator
==============================

This Python script is designed to calculate the total number
of days spent outside the UK from a flight history log, which
is particularly useful for UK Visa and Immigration (UKVI)
applications like Indefinite Leave to Remain (ILR) that have
limits on absences.

.. note:: Generated with Google Gemini

"""
# Libraries
import pandas as pd


def calculate_time_abroad_from_df(
        df: pd.DataFrame,
        date_col: str,
        status_col: str,
        dep_port_col: str,
        arr_port_col: str
) -> tuple[pd.Timedelta, pd.DataFrame]:
    """
    Calculates total time spent abroad from a flight log in a pandas DataFrame,
    handling multi-airport routes.

    Args:
        df: The DataFrame containing the flight log.
        date_col: The name of the column with the flight timestamps.
        status_col: The name of the column with the 'Inbound'/'Outbound' status.
        dep_port_col: The name of the departure port column.
        arr_port_col: The name of the arrival port column.

    Returns:
        A tuple containing:
        - The total duration abroad (pandas.Timedelta).
        - A new DataFrame with details of the valid round trips, including routes.
    """

    # --- 1. Prepare the DataFrame ---
    df_processed = df.copy()
    df_processed[date_col] = pd.to_datetime(
        df_processed[date_col], dayfirst=True, errors='coerce')
    df_processed = df_processed.dropna(subset=[date_col]) \
        .sort_values(by=date_col).reset_index(drop=True)

    # --- 2. Calculate Time Abroad with State Machine Logic ---
    total_duration_abroad = pd.Timedelta(0)
    last_outbound_leg = None  # Stores the entire row of the last outbound flight
    valid_trips = []

    print("--- Processing Flight Log ---")

    for index, row in df_processed.iterrows():
        if row[status_col] == 'Outbound':
            if last_outbound_leg is not None:
                print(f"⚠️ Warning: Consecutive Outbound flight on {row[date_col]}. Using this as the new departure.")
            last_outbound_leg = row

        elif row[status_col] == 'Inbound':
            if last_outbound_leg is not None:
                # A valid return trip is found
                duration = row[date_col] - last_outbound_leg[date_col]
                total_duration_abroad += duration


                # Capture the outbound and inbound journeys separately
                outbound_route = f"{last_outbound_leg[dep_port_col]} → {last_outbound_leg[arr_port_col]}"
                inbound_route = f"{row[dep_port_col]} → {row[arr_port_col]}"

                # Combine them for a full description
                full_route_str = f"Out: {outbound_route} | In: {inbound_route}"

                valid_trips.append({
                    'Outbound Date': last_outbound_leg[date_col],
                    'Inbound Date': row[date_col],
                    'Duration (days)': duration.days,
                    'Route': full_route_str
                })

                # Reset state to "Home"
                last_outbound_leg = None
            else:
                print(
                    f"⚠️ Warning: Ignoring Inbound flight on {row[date_col]} as there was no preceding Outbound flight.")

    if last_outbound_leg is not None:
        print(
            f"\n⚠️ Note: Log ends while abroad. The last trip starting {last_outbound_leg[date_col]} is not included.")

    trips_df = pd.DataFrame(valid_trips)

    return total_duration_abroad, trips_df







# ------------------------------------------
# Main
# ------------------------------------------
# Libraries
from pathlib import Path

# Configuration
id = '775243'
id = '1085721'
out_path = Path(f'./outputs/{id}')

# Load data
flight_df = pd.read_json(out_path / 'flights.json')

# 2. Call the function with all the required column names
total_time, valid_trips_df = calculate_time_abroad_from_df(
    df=flight_df,
    date_col='Departure Date/Time',
    status_col='In/Out',
    dep_port_col='Dep Port',
    arr_port_col='Arrival Port'
)

# 3. Print the final results
print("\n--- Final Summary ---")
print(f"✅ Total time spent abroad: {total_time.days} days (exact duration)")
print(f"✅ Total time spent abroad: %s days (full days only)" % \
    valid_trips_df['Duration (days)'].sum())

print("\n--- Valid Round Trips ---")
print(valid_trips_df)
