"""
UKVI trips visualisation
-----------------------------

"""

import pdfplumber
import re
import json
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter
from datetime import datetime
from pathlib import Path




def extract_basic_travel_data(pdf_path, start_page, end_page):
    """Extract the data from a PDF file.

    Ensure that the format is appropriate, ent the column headers match those
    included below. Otherwise modify as appropriate.

    Parameters
    ----------
    pdf_path: str
        The path to the file.
    start_page: int
        The start page where the table apperas.
    end_page: int
        The end page where the table appears.

    Returns
    -------
    """
    # Define the headers for essential data
    headers = [
        "Departure Date/Time", "Arrival Date/Time", "Voyage Code", "In/Out",
        "Dep Port", "Arrival Port"
    ]

    travel_data = []

    # Regex pattern to capture essential information
    row_pattern = re.compile(
        r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2})\s+"  # Departure Date/Time
        r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2})\s+"  # Arrival Date/Time
        r"(\S+)\s+"                            # Voyage Code
        r"(Outbound|Inbound)\s+"               # In/Out
        r"(\S+)\s+"                            # Dep Port
        r"(\S+)"                               # Arrival Port
    )

    # Open the PDF file and iterate over specified pages
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(start_page - 1, end_page):
            page = pdf.pages[page_num]
            text = page.extract_text()
            if not text:
                continue

            # Match rows using the regex pattern
            matches = row_pattern.findall(text)
            if matches:
                for match in matches:
                    travel_data.append(dict(zip(headers, match)))

    # Return
    return travel_data


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

    # Return
    return pd.DataFrame(result)



def display(df, cmap=None):
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
            cmap.get(row['Outbound Ports'].split('-')[1], 'skyblue')


        # Plot each voyage as a horizontal bar with text annotations
        ax.plot([row["Outbound Date"], row["Inbound Date"]], [i, i], marker='o', color=color, lw=6)
       
        # Formatting outbound and inbound dates
        outbound_str = row["Outbound Date"].strftime("%d %b")  # Day and abbreviated month
        inbound_str = row["Inbound Date"].strftime("%d %b")    # Day and abbreviated month

        # Adjust the text position to be further right
        ax.text(row["Inbound Date"] + pd.Timedelta(days=10), i - 0.05,  # Increased offset to 10 days
                f"{row['Outbound Ports']} ({outbound_str}) to {row['Inbound Ports']} ({inbound_str}) | {row['Days Difference']} days", 
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
    years = pd.date_range(start=start_date, end=end_date+pd.DateOffset(years=1), freq='Y')
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
    plt.show()



# ------------------------------------------------------
# Main
# ------------------------------------------------------
# Include any missing entry. This could happen if the travel
# was done by bus or train, as only flights have been recorded
# in the system.
MISSING = {
    'veronica': [
        {
            "Departure Date/Time": "22/04/2019 18:21",
            "Arrival Date/Time": "22/04/2019 20:21",
            "Voyage Code": "BUS001",
            "In/Out": "Inbound",
            "Dep Port": "AMS",
            "Arrival Port": "LDN"
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



# Define the PDF file path and page range to extract
pdf_path = Path('./data/775243 Final Bundle.pdf')
start_page = 6  # Page number where the tables start
end_page = 8    # Page number where the tables end

# Define the JSON file
#pdf_path = Path('./data/bernard-2024.json')

# Load DataFrame
if pdf_path.suffix == '.pdf':
    trips = extract_basic_travel_data(pdf_path, start_page, end_page)
elif pdf_path.suffix == '.json':
    trips = pd.read_json(pdf_path)
else:
    print('File extension <%s> not supported.' % pdf_path.suffix)

# Convert to DataFrame
df = pd.DataFrame(trips)
# Append missing rows using concat
df = pd.concat([df, pd.DataFrame(MISSING['veronica'])], ignore_index=True)
# Combine consecutive outbound-inbound trips into one row.
df_cmb = combine_outbound_inbound(df)

# Show
print(df_cmb)

# Display
display(df_cmb)