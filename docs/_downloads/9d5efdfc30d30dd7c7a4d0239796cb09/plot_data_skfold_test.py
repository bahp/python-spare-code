"""
01. Test Skfold
===============

Demonstrate that ``StratifiedKFold`` is deterministic and
always returns the same splits. The way to change the splits
is by changing the random state.
"""
# General
import numpy as np
import pandas as pd

# Specific
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------
#
# ---------------------------------------------------
def repeated_splits(X, y, n_loops=2, n_splits=5):
    """This method creates several times the
       splits using the same function. Then
       it is used to check that the splitting
       is always consistent.
    """
    # Record for comparison
    records = []

    # Split
    for i in range(n_loops):
        # Create dataframe
        dataframe = pd.DataFrame()
        # Create splitter
        skf = StratifiedKFold(n_splits=n_splits)
        # Loop
        for j, (train, test) in enumerate(skf.split(X, y)):
            dataframe['fold_{0}'.format(j)] = \
                np.concatenate((train, test))
        # Append
        records.append(dataframe)

    # Return
    return records


# ---------------------------------------------------
# Artificial example
# ---------------------------------------------------
# Size
n = 2000
n_splits = 5
n_loops = 5

# Create dataset
X = np.random.randint(10, size=(n, 7))
y = (np.random.rand(n) > 0.1).astype(int)

# Create splits
records = repeated_splits(X, y, n_loops=n_loops,
                                n_splits=n_splits)

# Compare if all records are equal
print("\nExample I:")
for i in range(len(records)-1):
    print('{0} == {1} : {2}'.format(i, i+1, \
        records[i].equals(records[i+1])))



# ---------------------------------------------------
# Real example
# ---------------------------------------------------
# Libraries
from sklearn.datasets import load_iris

# Load data
bunch = load_iris(as_frame=True)

# Label conversion
lblmap = dict(enumerate(bunch.target_names))

# Dataframe
df = bunch.data
df['target'] = bunch.target
df['label'] = df.target.map(lblmap)

# Get X, and y
X = df.to_numpy()
y = df.label.to_numpy()

# Create splits
records = repeated_splits(X, y, n_loops=n_loops,
                                n_splits=n_splits)

# Compare if all records are equal
print("\nExample II:")
for i in range(len(records)-1):
    print('{0} == {1} : {2}'.format(i, i+1, \
        records[i].equals(records[i+1])))

