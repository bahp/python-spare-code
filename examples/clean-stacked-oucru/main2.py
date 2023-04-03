"""
Clean oucru - main 02
=====================
"""

# Libraries
import pandas as pd

# -----------------------------
# Helper methods
# -----------------------------
def create_ids(series):
    """This method creates the ids"""
    # Unique patient numbers
    unique = series.unique()
    # Patient mapping
    mapping = dict(zip(unique, range(len(unique))))
    # Result (mysql pk ids from 1)
    aux = series.map(mapping) + 1
    # Return
    return aux

def keep_by_table(df, tablename, remove_prefix=True):
    """Keep columns starting with table name"""
    # Define tag
    tag = '%s_' % tablename
    # Find columns with tag
    cols = [c for c in df.columns if c.startswith(tag)]
    # Keep only those columns
    aux = df[cols].copy(deep=True)
    aux = aux.drop_duplicates()
    aux = aux.reset_index(drop=True)
    # Remove prefix
    if remove_prefix:
        aux.columns = [c.replace(tag, '') for c in cols]
    # Return
    return aux


# -----------------------------
# Load data
# -----------------------------
# Load data
data = pd.read_csv('./data/combined_stacked.csv',
    #nrows=1000,
    parse_dates=['date'])

# Show
print(data)
print(data.columns)

# -----------------------------
# Format
# -----------------------------
# Drop
drop = [
    'Unnamed: 0',
    'Unnamed: 0.1',
    'result_old',
    'date_old',
    'dsource'
]

# Rename
rename = {
    'study_no': 'patient_nhs_number',
    'date': 'date_collected',
    'column': 'laboratory_code',
}

# Replace
replace = {
    'result': {
        'True': 1,
        'False': 0
    }
}

# Basic format
data = data.drop(columns=drop)
data = data.rename(columns=rename)

# Boolean to number
data = data.replace(replace)

# Keep only those whose result can be cast to number
data.result = pd.to_numeric(data.result, errors='coerce')

# These columns are required
data = data.dropna(how='any',
    subset=['date_collected', 'result'])

# Reset index
data = data.reset_index()

# Show types
print(data.dtypes)



# ---------------------------------
# Create patients
# ---------------------------------
# Create ids
data['patient_id'] = create_ids(data.patient_nhs_number)

# Create DataFrame
patient = keep_by_table(data, tablename='patient')

# Add dates
patient['date_created'] = pd.to_datetime('today').normalize()
patient['date_updated'] = pd.to_datetime('today').normalize()
patient['name'] = ''
patient['surname'] = ''
patient['ext_number'] = patient.nhs_number # can be null or unique
patient['hos_number'] = patient.nhs_number # can be null or unique
patient['dob'] = pd.to_datetime('today').normalize()      # can be null
patient['gender'] = 0

# Order
patient = patient[['id',
                   'date_created',
                   'date_updated',
                   'name',
                   'surname',
                   'ext_number',
                   'nhs_number',
                   'hos_number',
                   'dob',
                   'gender']]

# Show
print("\nPatient table:")
print(patient)

# ---------------------------------
# Create laboratory code
# ---------------------------------
# Create ids
data['laboratory_id'] = create_ids(data.laboratory_code)

# Create DataFrame
laboratory_codes = \
    keep_by_table(data, tablename='laboratory')

# Add dates
laboratory_codes['date_created'] = pd.to_datetime('today').normalize()
laboratory_codes['date_updated'] = pd.to_datetime('today').normalize()
laboratory_codes['name'] = laboratory_codes.code
laboratory_codes['description'] = ''
laboratory_codes['is_visible'] = 1

# Order
laboratory_codes = laboratory_codes[['id',
                                     'date_created',
                                     'date_updated',
                                     'code',
                                     'name',
                                     'description',
                                     'is_visible']]

# Show
print("\nLaboratory codes table:")
print(laboratory_codes)

# ---------------------------------
# Create laboratory test
# ---------------------------------
# Drop columns
data = data.drop(columns=[
    'patient_nhs_number',
    'laboratory_code'
])

# Rename columns
data = data.rename(columns={
    'laboratory_id': 'code_id',
})

# Add columns
data['id'] = data.index.values + 1
data['uuid'] = data.id
data['date_outcome'] = data.date_collected
data['date_created'] = data.date_collected
data['date_updated'] = data.date_collected
data['unit_range'] = ''
data['result_status'] = 'UNKNOWN'
data['abnormal_status'] = ''

# Order
data = data[['id',
             'uuid',
             'date_created',
             'date_updated',
             'date_collected',
             'date_outcome',
             'result',
             'unit',
             'unit_range',
             'result_status',
             'abnormal_status',
             'code_id',
             'patient_id']]



# Show
print("\nLaboratory tests table:")
print(data)

# ---------------------------------
# Save
# ---------------------------------
# Save
patient.to_csv('./outputs/patient.csv',
    date_format='%Y-%m-%d %H:%M:%S', index=False)
laboratory_codes.to_csv('./outputs/laboratory_code.csv',
    date_format='%Y-%m-%d %H:%M:%S', index=False)
data.to_csv('./outputs/laboratory_tests.csv',
    date_format='%Y-%m-%d %H:%M:%S', index=False)


