# Libraries
import pandas as pd

# Load data
df = pd.read_csv('./data/kaggle120/athlete_events.csv')

from geopy.geocoders import Nominatim
import pandas as pd
import time  # To avoid hitting the API too frequently

# Initialize the geolocator
geolocator = Nominatim(user_agent="olympics")

# Function to get country from city name
def city_to_country(city_name):
    try:
        location = geolocator.geocode(city_name)
        if location:
            address = location.raw.get('address', {})
            print(address)
            return address.get('country', None)
        return None
    except Exception as e:
        print(f"Error occurred for city {city_name}: {e}")
        return None


df = df.head(2)

# Apply the function to the DataFrame
df['Country'] = df['City'].apply(city_to_country)

# To avoid hitting the API too frequently, you can add a sleep interval
# Uncomment the following line to add a delay
# time.sleep(1)  # Sleep for 1 second

print(df)


import sys
sys.exit()

print(df)
print(df.info)
print(df.columns.tolist())
print(df.Medal.unique())
# Create DataFrame

medal_counts = df.groupby(['Year', 'Season'])['Medal'] \
    .value_counts().unstack(fill_value=0).reset_index()

print(medal_counts)