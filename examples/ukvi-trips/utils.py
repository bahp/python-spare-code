import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter
from pathlib import Path


"""
 {
            "Departure Date/Time":"05/07/2022 15:35",
            "Arrival Date/Time":"05/07/2022 16:55",
            "Voyage Code":"FR0124",
            "In/Out":"Outbound",
            "Dep Port":"STN",
            "Arrival Port":"AOI"
        },
        {
            "Departure Date/Time":"08/02/2021 15:35",
            "Arrival Date/Time":"08/02/2021 16:55",
            "Voyage Code":"IB3162",
            "In/Out":"Inbound",
            "Dep Port":"MAD",
            "Arrival Port":"LHR"
        }

{
   "Departure Date/Time":"15/11/2022 21:25",
   "Arrival Date/Time":"15/11/2022 22:55",
   "Voyage Code":"MH0001",
   "In/Out":"Outbound",
   "Dep Port":"LHR",
   "Arrival Port":"KUL"
},
"""

# ----------------------------------------------
# Constants
# ----------------------------------------------
# Include any missing entry. This could happen if the travel
# was done by bus or train, as only flights have been recorded
# in the system.
MISSING_FLIGHTS = {
    '775243': [
        {
            "Departure Date/Time": "22/04/2019 18:21",
            "Arrival Date/Time": "22/04/2019 20:21",
            "Voyage Code": "BUS001",
            "In/Out": "Inbound",
            "Dep Port": "AMS",
            "Arrival Port": "LDN"
        }
    ],
    '1085721': [
        {
            "Departure Date/Time":"08/02/2021 15:35",
            "Arrival Date/Time":"08/02/2021 16:55",
            "Voyage Code":"IB3162",
            "In/Out":"Inbound",
            "Dep Port":"MAD",
            "Arrival Port":"LHR"
        },
        {
            "Departure Date/Time":"05/07/2022 15:35",
            "Arrival Date/Time":"05/07/2022 16:55",
            "Voyage Code":"FR0124",
            "In/Out":"Outbound",
            "Dep Port":"STN",
            "Arrival Port":"AOI"
        }
    ]
}

# Include the colors desired for each airport. For example
# they could be colored by country.
COLORMAP = {
    'FRA': 'black',
    'BLQ': 'green',
    'LHR': 'blue',
    'LGW': 'blue',
    'RAK': 'skyblue',
    'STN': 'blue',
    'AOI': 'green',
    'MXP': 'green',
    'LTN': 'blue',
    'JNB': 'skyblue',
    'AMS': 'skyblue',
    'BGY': 'green',
    'ZRH': 'skyblue',
    'ALC': 'yellow',
    'TFS': 'yellow',
    'BSL': 'skyblue',
    'MAD': 'yellow',
    'VRN': 'green',
    'ATH': 'skyblue',
    'LPA': 'yellow',
    'FCO': 'green',
    'FRFHN': 'black',
    'LDN': 'blue'
}

# ----------------------------------------------
# Methods
# ----------------------------------------------
def combine_outbound_inbound(df):
    """Combine Outbound-Ibound rows into a single one.

    Paramters
    ---------
    df: pd.DataFrame
        The DataFrame with the data.

    Returns
    -------
    pd.DataFrame

    """

    # Convert date columns to datetime format
    df["Departure Date/Time"] = \
        pd.to_datetime(df["Departure Date/Time"],
            format="%d/%m/%Y %H:%M")
    df["Arrival Date/Time"] = \
        pd.to_datetime(df["Arrival Date/Time"],
            format="%d/%m/%Y %H:%M")

    # Sort the DataFrame by "Departure Date/Time"
    df = df.sort_values(by="Departure Date/Time").reset_index(drop=True)

    """
    # Process the DataFrame
    result = []
    for i in range(0, len(df) - 1, 2):  # Step by 2 to handle consecutive rows
        outbound = df.iloc[i]
        inbound = df.iloc[i + 1]

        # Ensure the pair consists of an outbound followed by an inbound
        if outbound["In/Out"] == "Outbound" and inbound["In/Out"] == "Inbound":
            # Calculate the difference in days
            days_difference = (inbound["Arrival Date/Time"] - outbound["Departure Date/Time"]).days - 1

            # Create a combined row with desired columns
            combined_row = {
                "Outbound Date": outbound["Departure Date/Time"],
                "Inbound Date": inbound["Arrival Date/Time"],
                "Outbound Ports": outbound["Dep Port"] + '-' + outbound["Arrival Port"],
                "Inbound Ports": inbound["Dep Port"] + '-' + inbound["Arrival Port"],
                "Days Difference": days_difference,
                "Voyage Code": outbound["Voyage Code"]
            }

            result.append(combined_row)
    """

    result = []
    # This variable will store the last seen outbound flight
    outbound_leg = None

    # Iterate through the DataFrame one row at a time
    for index, current_flight in df.iterrows():

        # Case 1: The current flight is an Outbound flight
        if current_flight["In/Out"] == "Outbound":
            # If we see a new outbound flight, it becomes the start of our current trip.
            # This handles consecutive 'Outbound' flights by just taking the latest one.
            outbound_leg = current_flight

        # Case 2: The current flight is an Inbound flight
        elif current_flight["In/Out"] == "Inbound":
            # Check if we have a saved outbound flight to pair it with
            if outbound_leg is not None:
                # We have a valid pair!

                # Calculate the difference in days
                days_difference = (current_flight["Arrival Date/Time"] - outbound_leg["Departure Date/Time"]).days

                # Create the combined row
                combined_row = {
                    "Outbound Date": outbound_leg["Departure Date/Time"],
                    "Inbound Date": current_flight["Arrival Date/Time"],
                    "Outbound Ports": f"{outbound_leg['Dep Port']}-{outbound_leg['Arrival Port']}",
                    "Inbound Ports": f"{current_flight['Dep Port']}-{current_flight['Arrival Port']}",
                    "Days Difference": days_difference,
                    "Voyage Code": outbound_leg["Voyage Code"]
                }
                result.append(combined_row)

                # CRITICAL: Reset outbound_leg to None since the trip is now complete.
                # We are now ready to find the next outbound flight.
                outbound_leg = None

            # If outbound_leg is None here, it means we found an inbound flight
            # without a preceding outbound flight, so we just ignore it.
    # Return
    return pd.DataFrame(result)


def display_flights(df, cmap=None):
    """Plotting the graph.

    Parameters
    ----------
    df: pd.DataFrame
        The pandas DataFrame.

    Returns
    -------
    None
    """
    # Set up plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Fore each row (voyage)
    for i, row in df.iterrows():

        if cmap is None:
            color = 'skyblue'
        else:
            color = cmap.get(row['Outbound Ports'].split('-')[1], 'skyblue')


        # Plot each voyage as a horizontal bar with text annotations
        ax.plot([row["Outbound Date"], row["Inbound Date"]], [i, i], marker='o', color=color, lw=6)

        # Formatting outbound and inbound dates
        outbound_str = row["Outbound Date"].strftime("%d %b")  # Day and abbreviated month
        inbound_str = row["Inbound Date"].strftime("%d %b")    # Day and abbreviated month

        # Adjust the text position to be further right
        ax.text(row["Inbound Date"] + pd.Timedelta(days=10), i - 0.05,  # Increased offset to 10 days
                f"{row['Outbound Ports']} ({outbound_str}) → {row['Inbound Ports']} ({inbound_str}) | {row['Days Difference']} days",
                va='center', ha='left', fontsize=9, color="black")

    # Alternate month shading
    start_date = df["Outbound Date"].min().replace(day=1)
    end_date = df["Inbound Date"].max()
    current_date = start_date
    month = 0
    while current_date < end_date:
        next_month = (current_date + pd.DateOffset(months=1)).replace(day=1)
        ax.axvspan(current_date, next_month, color='gray' if month % 2 == 0 else 'lightgray', alpha=0.2)
        current_date = next_month
        month += 1

    # Add horizontal lines for each year
    years = pd.date_range(start=start_date, end=end_date+pd.DateOffset(years=1), freq='YE')
    for year in years:
        ax.axvline(year, color='black', linestyle='--', lw=1)  # Vertical line for each year
        ax.text(year - pd.Timedelta(days=90), len(df) + 0.5, year.year,
            ha='left', va='center', fontsize=10, color='black')  # Year label

    # Setting the x-axis limits to include full years
    full_start_date = pd.Timestamp(year=start_date.year, month=1, day=1)
    full_end_date = pd.Timestamp(year=end_date.year + 1, month=1, day=1)  # Next January
    ax.set_xlim(full_start_date, full_end_date)

    # Set x-axis ticks to show full years from January to December
    ax.xaxis.set_major_locator(mdates.YearLocator())   # Major ticks at the beginning of each year
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for each month
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))  # Year as the format for major ticks

    # Formatting the plot
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Voyage Code"])
    #ax.set_yticklabels(df['Days Difference'])
    ax.set_xlabel("Date")
    ax.set_title("Voyage Durations (total abroad %s days)" % df['Days Difference'].sum())

    # Set x-axis ticks to show abbreviated month names and year
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))  # Month abbreviation and year
    plt.xticks(rotation=45)

    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()



