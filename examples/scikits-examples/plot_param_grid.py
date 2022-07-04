"""
ParamGrid
============

This example shows different configurations of ParamGrid.
"""

# Generic
import pandas as pd

# Library
from sklearn.model_selection import ParameterGrid

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False

# -----------------------------
# Sample I
# -----------------------------
# Define params
params = {'a': [1, 2], 'b': [True, False]}
# Create params grid
grid = list(ParameterGrid(params))
# Show
if TERMINAL:
    print("\nExample I:")
    print(pd.DataFrame(grid))
pd.DataFrame(grid)

# %%
#


# -----------------------------
# Sample II
# -----------------------------
# Define params
params = [{'kernel': ['linear']},
          {'kernel': ['rbf'], 'gamma': [1, 10]}]
# Create params grid
grid = list(ParameterGrid(params))
# Show
if TERMINAL:
    print("\nExample II:")
    print(pd.DataFrame(grid))
pd.DataFrame(grid)

# %%
#

# -----------------------------
# Sample III
# -----------------------------
# Define params
params = {'imp': ['simp', 'iimp'],
          'scl': ['minmax', 'norm', 'std'],
          'est': ['PCA', 'AE']}
# Create params grid
grid = list(ParameterGrid(params))
# Show
if TERMINAL:
    print("\nExample III:")
    print(pd.DataFrame(grid))
pd.DataFrame(grid)

# %%
#


# -----------------------------
# Sample IV
# -----------------------------
# Define params
params = [{'imp': ['simp', 'iimp'], 'scl': ['minmax'], 'est': ['PCA']},
          {'imp': ['simp', 'iimp'], 'scl': ['std', 'norm'], 'est': ['AE']}]
# Create params grid
grid = list(ParameterGrid(params))
# Show
if TERMINAL:
    print("\nExample IV:")
    print(pd.DataFrame(grid))
pd.DataFrame(grid)

# %%
#