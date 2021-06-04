"""
Pandas - Main 01
================
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
# Date will be a datetime65[ns] instead of string
data.date = pd.to_datetime(data.date, dayfirst=True)
data.date = data.date.dt.normalize()

# Show
if TERMINAL:
    print("\nData:")
    print(data)
data

####################################################################
# Sort values

# Note that if you set columns as indexes (e.g. the
# datetime) they will be sorted by default.
aux = data.sort_values(by=['plt', 'hct'])

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux

###################################################################
# Select columns

# Select columns from DataFrame
aux = data[['patient', 'date', 'plt']]

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux

###################################################################
# Indexing (not nan)
#

# Keep rows where plt is not nan
aux = data[data.plt.notna()]

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux


###################################################################
# Drop nan (in subset)
#

# Keep rows without any nan in subset
aux = data.dropna(how='any', subset=['plt', 'bil'])

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux


###################################################################
# Drop nan (all)
#

# Keep rows without any nan at all
aux = data.dropna(how='any')

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux

###################################################################
# Resample dates
#
# Resample
aux = data.set_index('date').resample('D').asfreq()

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux

###################################################################
# Filling missing (pad)
#

# Pad is synonym of DataFrame.fillna() with method='ffill'.
aux = data.set_index('date').resample('D').asfreq().pad()

# Show
if TERMINAL:
    print("\nOut:")
    print(aux)
aux