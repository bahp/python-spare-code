"""
01. Sliding window
==================
"""

# Interesting code.
# np.lib.stride_tricks.sliding_window_view(df.index, 3)

# Libraries
import numpy as np
import pandas as pd

# Configuration
ROWS, COLS = 1500, 10
PATIENTS = 100

# Create random values
features = np.random.random_sample((ROWS, COLS))
patients = np.random.randint(PATIENTS, size=(ROWS,1))

# Create DataFrame
df = pd.DataFrame(data=features)
df = df.add_prefix('feature_')
df['patient'] = patients
df['day'] = -(df.groupby('patient').cumcount()+1)
df = df.sort_values(by=['patient', 'day'],
    ascending=[True, True]).reset_index(drop=True)

# Show
print("\nData:")
print(df)

# ----------------------------------
# Method I: Own method
# ----------------------------------
def sliding_window_iter(series, size, include_id=True):
    """series is a column of a DataFrame.

    .. note: The DataFrame should be pre-ordered to ensure
             that IDs remain consistent.
    """
    for i, start_row in enumerate(range(len(series) - size + 1)):
        s = series[start_row:start_row + size]
        if include_id:
            s['window'] = i
        yield s


# Group by patient and compute sliding window
result = df.groupby('patient').apply(lambda x:
    pd.concat(sliding_window_iter(x, 3)))

# Show
print("\nResult:")
print(result)

# ----------------------------------
# Method II: Using rolling
# ----------------------------------
#a = df.groupby('patient').rolling(window=3)
#b = [win for win in a if win.shape[0] == 3]
#c = pd.concat(b)
#print(c)