"""
Clean oucru - main 01
=====================

"""
# Libraries
import pandas as pd

# -----------------------------
# Load data
# -----------------------------
# Load data
data = pd.read_csv('./data/combined_stacked.csv',
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
    'study_no' : 'patient',
    'date': 'date_collected',
    'column': 'code',
}

# Replace
replace = {
    'result': {
        'True': 1,
        'False': 0
    }
}

# Format
data = data.drop(columns=drop)
data = data.rename(columns=rename)
data['id'] = data.index.values
data['uuid'] = data.index.values
data['date_outcome'] = data.date_collected
data = data.replace(replace)
data = data[sorted(data.columns.values)]

# Keep only those whose result can be cast to number
data.result = pd.to_numeric(data.result, errors='coerce')

# Remove nan
data = data[data.result.notna()]

# Show types
print(data.dtypes)

# Save
data.head(10000).to_csv('./outputs/combined_stack_head10000.csv')
data.to_csv('./outputs/combined_stacked.csv')
