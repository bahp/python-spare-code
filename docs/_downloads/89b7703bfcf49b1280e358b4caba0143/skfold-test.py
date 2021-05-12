"""
Skfold Test
=============

Example
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
    """This method...


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
n = 20
n_splits = 5
n_loops = 5

# Create dataset
X = np.arange(n).reshape(-1,1)
y = np.vstack((np.ones((10,1)),
               np.zeros((10,1))))

# Create splits
records = repeated_splits(X, y, n_loops=n_loops,
                                n_splits=n_splits)

# Compare if all records are equal
for i in range(len(records)-1):
    print('{0} == {1} : {2}'.format(i, i+1, \
        records[i].equals(records[i+1])))


# ---------------------------------------------------
# Rel example
# ---------------------------------------------------
# Read dataset
dataset = pd.read_csv('dataset.csv')

# Get X, and y
X = dataset.to_numpy()
y = dataset.micro_confirmed.to_numpy()

# Create splits
records = repeated_splits(X, y, n_loops=n_loops,
                                n_splits=n_splits)

# Compare if all records are equal
for i in range(len(records)-1):
    print('{0} == {1} : {2}'.format(i, i+1, \
        records[i].equals(records[i+1])))

