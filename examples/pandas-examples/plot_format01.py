#
import numpy as np
import pandas as pd

ROWS, COLS = 20, 10

# Create random values
features = np.random.random_sample((ROWS, COLS))
patients = np.random.randint(5, size=(ROWS,1))
windows = np.random.randint(2, size=(ROWS,1))
#matrix = np.hstack((patients, windows, features))

# Create DataFrame
df = pd.DataFrame(data=features)
df = df.add_prefix('feature_')
df['patient'] = patients
#df['window'] = windows
df['day'] = -(df.groupby('patient').cumcount()+1)
df = df.sort_values(by=['patient', 'day'],
                    ascending=[True, True]) \
    .reset_index(drop=True)

# Show
print("\nData")
print(df)


np.lib.stride_tricks.sliding_window_view(df.index, 3)


df['aux'] = df.groupby('patient') \
    .rolling(on='day', window=3, min_periods=3) \
    .apply(lambda x: 1)

print("\n\n\n\n\n\n")
df
"""
df = df.groupby('patient') \
    .rolling(window=5, axis=1) \
    .apply(f)
"""
