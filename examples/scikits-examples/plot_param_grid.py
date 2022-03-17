# Generic
import pandas as pd

# Library
from sklearn.model_selection import ParameterGrid

# -----------------------------
# Sample I
# -----------------------------
# Define params
params = {'a': [1, 2], 'b': [True, False]}
# Create params grid
grid = list(ParameterGrid(params))
# Show
print("\nExample I:")
print(pd.DataFrame(grid))



# -----------------------------
# Sample II
# -----------------------------
# Define params
params = [{'kernel': ['linear']},
          {'kernel': ['rbf'], 'gamma': [1, 10]}]
# Create params grid
grid = list(ParameterGrid(params))
# Show
print("\nExample II:")
print(pd.DataFrame(grid))



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
print("\nExample IV:")
print(pd.DataFrame(grid))



# -----------------------------
# Sample III
# -----------------------------
# Define params
params = [{'imp': ['simp', 'iimp'], 'scl': ['minmax'], 'est': ['PCA']},
          {'imp': ['simp', 'iimp'], 'scl': ['std', 'norm'], 'est': ['AE']}]
# Create params grid
grid = list(ParameterGrid(params))
# Show
print("\nExample III:")
print(pd.DataFrame(grid))