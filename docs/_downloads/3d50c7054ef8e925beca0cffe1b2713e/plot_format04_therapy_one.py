"""
04. Transforming Patient Therapy Intervals to a Daily Summary
=============================================================

Previous title: 04. Format MIMIC therapy (one)

This script demonstrates how to process time-interval data, a common format in
medical records like the MIMIC-III dataset. It starts with a table where each
row represents an antibiotic therapy with a specific start and end date.

The main goal is to transform this interval-based data into a daily summary.
This is achieved by using `pd.date_range` to generate all dates within each
therapy period and then using the `explode()` method to create a distinct row
for each day. The final output is a count of how many days each unique
combination of therapies was administered to the patient.

"""
# Generic libraries
import pandas as pd

# Show in terminal
TERMINAL = False

##################################################################
# First, lets load and do some basic formatting on the data.

# -----------------------------
# Constants
# -----------------------------
# Path
path = './data/mimic-therapy/One_patient_condensed_10656173.csv'
path = './data/mimic-therapy/One_patient_condensed_11803145.csv'

# -----------------------------
# Load data
# -----------------------------
# Read data
data = pd.read_csv(path,
    dayfirst=True,
    parse_dates=['starttime',
                 'stoptime'])

# Keep only useful columns
data = data[['subject_id',
             'antibiotic',
             'route',
             'starttime',
             'stoptime']]

# Reformat (ignore time information)
data.starttime = data.starttime.dt.date
data.stoptime = data.stoptime.dt.date

# Show
if TERMINAL:
    print("\nData:")
    print(data)
data

####################################################################
# Lets transform the data

# -----------------------------
# Transform data
# -----------------------------
# .. note: The closed parameter indicates whether to include
#          the first and/or last samples. None will keep both,
#          left will keep only start date and right will keep
#          also the right date.
# Create column with date range
data['startdate'] = data.apply(lambda x:
    pd.date_range(start=x['starttime'],
                  end=x['stoptime'],
                  inclusive='left',         # ignoring right
                  freq='D') ,axis=1)

# Explode such column
data = data.explode('startdate')

# Create daily therapies
aux = data.groupby('startdate') \
    .apply(lambda x: sorted(x.antibiotic \
        .str.lower().str.strip().unique().tolist()))

# Include missing days
aux = aux.asfreq('1D')

####################################################################
# Lets see the formatted data

# Show
if TERMINAL:
    print("\nFormatted:")
    print(aux)
aux

####################################################################
# Lets count the number of days

# Show
if TERMINAL:
    print("\nTherapies (number of days)")
    print(aux.value_counts())
aux.value_counts()
