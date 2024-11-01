"""
99. Basic Example
==================
"""

# Library
import numpy as np
import pandas as pd

# Show in terminal
TERMINAL = False

# Create data
data = [
    ['p1', '1/5/2021', 1, 2, 3],
    ['p1', '2/5/2021', 3, 3, 3],
    ['p1', '3/5/2021', 4, 4, 4],
    ['p1', '5/5/2021', 5, 5, 5],

    ['p2', '11/5/2021', 5, 3, 3],
    ['p2', '12/5/2021', 4, 3, None],
    ['p2', '16/5/2021', None, 1, None], # unordered
    ['p2', '15/5/2021', 5, 2, 4],
]

# Load DataFrame
data = pd.DataFrame(data,
    columns=['patient', 'date', 'plt', 'hct', 'bil'])

# Format datetime
# Date will be a datetime64[ns] instead of string
data.date = pd.to_datetime(data.date, dayfirst=True)
data.date = data.date.dt.normalize()

# Show
if TERMINAL:
    print("\nData:")
    print(data)
data

####################################################################
# Lets sort values

# Note that if you set columns as indexes (e.g. the
# datetime) they will be sorted by default.
aux = data.sort_values(by=['plt', 'hct'])

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux

###################################################################
# Lets select columns

# Select columns from DataFrame
aux = data[['patient', 'date', 'plt']]

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux

###################################################################
# Lets do indexing (not nan)
#

# Keep rows where plt is not nan
aux = data[data.plt.notna()]

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux


###################################################################
# Lets drop nan (in subset)
#

# Keep rows without any nan in subset
aux = data.dropna(how='any', subset=['plt', 'bil'])

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux


###################################################################
# Lets drop nan (all)
#

# Keep rows without any nan at all
aux = data.dropna(how='any')

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux

###################################################################
# Lets resample daily
#

# Resample
aux = data.set_index('date').resample('D').asfreq()

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux

###################################################################
# Lets fill missing values (pad)
#

# Pad is synonym of DataFrame.fillna() with method='ffill'.
aux = data.set_index('date').resample('D').asfreq().pad()

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux

###################################################################
# Lets group by patient and sum

# Group by patient and sum
agg = aux.groupby('patient').sum()

# Show
if TERMINAL:
    print("\nOut:")
    print(agg)
agg

###################################################################
# Lets group by patient per 2 days and compute mean and max.

agg = aux.groupby(by=['patient', pd.Grouper(freq='2D')]) \
    .agg('mean', 'max')
    #.agg({'idx': ['first', 'last'],
    #      0: [skew, kurtosis, own],
    #      1: [skew, kurtosis, own],
    #      '0_hr': [own],
    #      '0_rr': [own]})

# Show
if TERMINAL:
    print("\nOut:")
    print(agg)
agg