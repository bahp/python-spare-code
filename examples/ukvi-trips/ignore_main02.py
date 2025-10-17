import pytesseract
import pandas as pd
from PIL import Image
import re
import json


# .. note: Replace with your own path

# Path to the uploaded image
image_path = './data/1085721-final-bundle/page_06.png'
# Perform OCR on the image
image = Image.open(image_path)

# Use custom Tesseract configurations to optimize OCR
custom_config = r'--psm 6'  # Treat image as a single block of text
raw_text = pytesseract.image_to_string(image, config=custom_config)

# Split the text into lines
lines = raw_text.split("\n")

# Define the desired headers for the DataFrame
headers = [
    "Departure Date/Time", "Arrival Date/Time", "Voyage Code", "In/Out",
    "Dep Port", "Arrival Port"
]

# Initialize a list to store the extracted rows
data = []

# Regular expression to match each row in the table
row_pattern = re.compile(
    r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2})\s+"  # Departure Date/Time
    r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2})\s+"  # Arrival Date/Time
    r"([A-Za-z0-9]+)\s+"                   # Voyage Code
    r"(Inbound|Outbound)\s+"               # In/Out
    r"([A-Z]{3})\s+"                       # Dep Port
    r"([A-Z]{3})"                          # Arrival Port
)

# Extract rows using the pattern
for line in lines:
    match = row_pattern.search(line)
    if match:
        data.append(match.groups())

# Create a DataFrame using the extracted data
df = pd.DataFrame(data, columns=headers)

# Export the DataFrame to JSON format with the specified structure
json_data = df.to_dict(orient="records")

# Save the JSON data to a file
json_file_path = "extracted_table.json"  # Replace with your desired path
with open(json_file_path, "w") as json_file:
    json.dump(json_data, json_file, indent=4)

# Print the output for confirmation
print(f"Extracted {len(df)} rows.")
print(f"JSON saved to {json_file_path}")
