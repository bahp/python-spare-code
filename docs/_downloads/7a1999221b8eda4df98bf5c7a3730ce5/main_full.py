"""
All patients
============

Example
"""
# Generic libraries
import pandas as pd

# -----------------------------
# Constants
# -----------------------------
# Path
path = './data/One_patient_condensed_10656173.csv'
path = './data/One_patient_condensed_11803145.csv'
#path = './data/ICU_diagnoses_antibiotics.csv'

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
             'hadm_id',
             'stay_id',
             'icd_code',
             'antibiotic',
             'route',
             'starttime',
             'stoptime']]

# Reformat (time info and str)
data.starttime = data.starttime.dt.date
data.stoptime = data.stoptime.dt.date
data.antibiotic = data.antibiotic \
    .str.lower() \
    .str.strip()

# Show
print("\nOriginal Data:")
print(data)

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
                  closed='left',         # ignoring right
                  freq='D') ,axis=1)

# Explode such column
data = data.explode('startdate')

# Groupby
groupby = ['subject_id',
           'hadm_id',
           'stay_id',
           'startdate']

# Create daily therapies
aux = data.groupby(groupby) \
    .apply(lambda x: sorted(x.antibiotic \
        .unique().tolist()))

# Show
print("\nFormatted:")
print(aux)
print("\nTherapies (number of days)")
print(aux.value_counts())
