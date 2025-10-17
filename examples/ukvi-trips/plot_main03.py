"""
UKVI Travel History Visualizer
==============================

This Python script is designed to automate the tracking and visualization
of international travel for UK immigration purposes. It extracts travel
data directly from a PDF report (such as a UKVI travel history record),
calculates the duration in days for each trip abroad, and generates a
clear timeline chart using Matplotlib. The resulting plot provides
an at-a-glance overview of all absences, making it easier to monitor
compliance with the continuous residence requirements for Indefinite
Leave to Remain (ILR) or citizenship applications. The script also
supports manual entry for trips not captured in the PDF and allows
for custom color-coding of destinations.

"""

import pytesseract
import pandas as pd
import json
import re

from PIL import Image
from pathlib import Path

# Set options to display all rows and columns#
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', None) # No truncation of cell content


def pdf2png(pdf_path, out_path, start_page=0, end_page=None, flag_save=True):
    """Converts the pdf to images and saves them in memory.

    Parameters
    ----------
    pdf_path: str
        The path for the pdf file.
    start_page: int
        The start page with tables.
    end_page: int
        The end page with tables.
    flag_save: bool
        Whether to save the images.

    Returns
    -------
    """
    # Libraries
    from pdf2image import convert_from_path

    # Convert images
    images = convert_from_path(pdf_path, first_page=start_page,
        last_page=end_page, dpi=600)

    # Save images
    if flag_save:
        if out_path is None:
            out_path = pdf_path.with_suffix('')
        out_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(out_path / ("page_%02d.png" % (i + start_page)))

    # Return
    return images


def png2json(png_path, out_path):
    """Processes PNG images to extract flight data and saves
    it to a JSON file.

    Parameters
    ----------
    pdf_path (pathlib.Path): The path to the original PDF file.

    Returns
    -------
    """
    # Extract all flights
    results = []
    for p in sorted(png_path.glob("*.png")):
        data = extract_basic_travel_data(p)
        results += data
        print(p)
        print(pd.DataFrame(data, columns=headers))
        print('\n\n')

    # Save results as json
    pd.DataFrame(results, columns=headers) \
        .to_json(out_path / 'flights.json',
            orient="records", indent=4)


def extract_basic_travel_data(image_path):
    """Extract the table information from an image.

    Parameters
    ----------
    image_path: str or Path
        The path with the image

    Returns
    -------
    """
    # Perform OCR on the image
    image = Image.open(image_path)
    custom_config = r'--psm 6'  # Assume uniform alignment for table
    raw_text = pytesseract.image_to_string(image, config=custom_config)

    # Initialize a list to store the extracted rows
    data = []

    # Regular expression to match each row in the table
    row_pattern = re.compile(
        r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2})\s+"  # Departure Date/Time
        r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2})\s+"  # Arrival Date/Time
        r"([A-Za-z0-9]+)\s+"  # Voyage Code
        r"(Inbound|Outbound)\s+"  # In/Out
        r"([A-Z]{3})\s+"  # Dep Port
        r"([A-Z]{3})"  # Arrival Port
    )

    # Process the raw text line by line
    lines = raw_text.split("\n")
    for i, line in enumerate(lines):
        match = row_pattern.search(line)
        if match:
            data.append(match.groups())

    # Return
    return data


def perform_validation(df):
    """"""
    MSG1 = 'Error: First flight in log is not Inbound'

    # Create a new column to store error messages, default to 'OK'
    df['validation_error'] = 'OK'

    # Rule 1: Check if the very first flight is Inbound
    if df.iloc[0]['In/Out'] != 'Inbound':
        df.loc[0, 'validation_error'] = MSG1

    # Rule 2: Check for consecutive Inbound/Outbound flights
    # This flags a row if it's the same as the one BEFORE it
    is_same_as_previous = df['In/Out'] == df['In/Out'].shift(1)
    # This flags a row if it's the same as the one AFTER it
    is_same_as_next = df['In/Out'] == df['In/Out'].shift(-1)

    # A row is an error if either of the above conditions is true
    is_consecutive = is_same_as_previous | is_same_as_next

    # Add error messages where the sequence is broken
    df.loc[is_consecutive, 'validation_error'] = \
        'Error: Part of consecutive ' + df['In/Out'] + ' pair'

    # Return
    return df




# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
# .. note:: Run the first time, later it can be disabled
#           since the extracting of pags from pdf to
#           png and the consequent OCR to extract the
#           that information which will be saved in a json file
#           only needs to be done once. Please check the json and
#           add any missing flights (either due to poor extraction
#           or not registered, or other methods of entry to the ountry"

# Flags
RUN_PDF2PNG = False  # Extract pdf pages to png images
RUN_OCR = False      # Extract data from png and save to json

# Define the desired headers for the DataFrame
headers = [
    "Departure Date/Time", "Arrival Date/Time",
    "Voyage Code", "In/Out", "Dep Port", "Arrival Port"
]

# Configuration of bundles.
config = {
    '1085721': {            # BH
        'start_page': 6,
        'end_page': 12
    },
    '775243': {             # VQ
        'start_page': 6,
        'end_page': 7
    }
}

# Select the bundle identifier
#id = '775243'
id = '1085721'

# Path
pdf_path = Path('./data/%s-final-bundle.pdf' % id)
out_path = Path('./outputs/%s' % id)

# Extract images from pdf
if RUN_PDF2PNG:
    pdf2png(pdf_path=pdf_path, out_path=out_path , **config[id])

# Extract all flights
if RUN_OCR:
    png2json(png_path=out_path, out_path=out_path)


# -----------------------------------------------
# Clean and display
# -----------------------------------------------
# Libraries
from utils import display_flights
from utils import combine_outbound_inbound
from utils import MISSING_FLIGHTS
from utils import COLORMAP

# Load DataFrame (as extracted)
df = pd.read_json(out_path / 'flights.json')

# Append missing rows using concat
df_miss = pd.DataFrame(MISSING_FLIGHTS[id])
df = pd.concat([df, df_miss], ignore_index=True)
df = df.drop_duplicates()

# Save results as json
df.to_json(out_path / 'flights.json',
    orient="records", indent=4)

# Ensure 'Departure Date/Time' is in datetime format
df["Departure Date/Time"] = pd.to_datetime(
    df["Departure Date/Time"], format="%d/%m/%Y %H:%M")

# Order chronologically
df = df.sort_values(by='Departure Date/Time').reset_index(drop=True)

# Remove duplicates based on the date (ignoring hour) and
# keep the first occurrence
df["Date Only"] = df["Departure Date/Time"].dt.date
df = df.drop_duplicates(subset=['Date Only', 'Voyage Code'], keep='first')

# Sort the DataFrame by "Departure Date/Time"
df = df.sort_values(by="Departure Date/Time").reset_index(drop=True)

# Show
print("\n\n---> Extracted flight history:\n\n%s" % df)


# Validate
# --------
# Perform validation
df = perform_validation(df)

# Extract errors
error_df = df[df['validation_error'] != 'OK']

print("\n\n")
if error_df.empty:
    print("All flight sequences are valid! ✅")
else:
    print("Found errors in the flight sequence: ⚠️ \n")
    print(error_df)



# Find the first 'Outbound'
# ------------------------
#  .. note:: This step has been made redundant. Its
#            functionality is now incorporated into the
#             combine_outbound_inbound function.

"""
# Find the first 'Outbound' flight and trim the DataFrame.
# We do this because we want to compute time abroad, hence
# each period would be (current outbound - next inbound)
try:
    # Get the index of the first row where 'In/Out' is 'Inbound'
    first_inbound_index = df[df['In/Out'] == 'Outbound'].index[0]
    # Slice the DataFrame to start from that index
    df = df.loc[first_inbound_index:].reset_index(drop=True)
except IndexError:
    pass
"""



# Combine in/out journeys
# -----------------------
# .. note:: For the most accurate results, this function should
#           be run after the flight data JSON has been manually
#           corrected. If the data is inconsistent, it will apply
#           its own assumptions to handle errors (e.g., pairing the
#           last seen 'Outbound' with the next 'Inbound' and
#           ignoring invalid sequences).

# Combine (inbound, outbound) pairs
df_cmb = combine_outbound_inbound(df)

# Save results as json
df_cmb.to_json(out_path / 'roundtrips.json',
    orient="records", indent=4)

# Show
print("\n\n---> Combined trips:\n\n%s" % df_cmb)


# Display and save
# -----------------------
import matplotlib.pyplot as plt
display_flights(df_cmb, cmap=COLORMAP)
plt.savefig(out_path / 'graph.jpg')
plt.show()
