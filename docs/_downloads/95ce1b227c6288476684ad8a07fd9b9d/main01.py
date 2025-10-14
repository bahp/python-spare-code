"""
01. Basic example
=================

Basic example to use tpot.
"""

# Import
import numpy as np
import pandas as pd

# Specific
from tpot import TPOTClassifier

# Import own
from pySML2.preprocessing.splitters import cvs_hos_split
from pySML2.preprocessing.splitters import kfolds_split


# ---------------------------------------------
# Configuration
# ---------------------------------------------
# The input features and label for the algorithm
features = sorted(['alb', 'alp', 'alt', 'baso', 'bil', 'cl', 'cre', 'crp', 'egfr',
                  'eos', 'k', 'ly',  'mcv', 'mono', 'mpv', 'nrbca', 'plt', 'rbc',
                  'rdw',  'urea', 'wbc'])

# The labels
labels = sorted(['micro_confirmed'])

# The splits
n_splits = 10

# Dataset
# -------
# Dataset filepath
filepath = 'data/dataset.csv'

# ---------------------------------------------
# Load dataset and format it
# ---------------------------------------------
# Read data
data = pd.read_csv(filepath)
data.columns = [c.lower() for c in data.columns.values]
# data = data[features + labels]

# Missing values
data['missing'] = data[features].isnull().sum(axis=1)

# The indexes for complete profiles
cmp = (data.missing == 0)

# Split in CVS and HOS
data['cvs_hos_split'] = cvs_hos_split(data, selected_rows=cmp)

# ---------------------------------------------
# Train
# ---------------------------------------------
data[(data.missing == 0)].to_csv('outcomes/main01/tpot_data_cvs.csv')
data[(data.cvs_hos_split == 'hos')].to_csv('outcomes/main01/tpot_data_hos.csv')
data[(data.cvs_hos_split == 'cvs')].to_csv('outcomes/main01/tpot_data_cvs.csv')
data[(data.cvs_hos_split == 'hos')].to_csv('outcomes/main01/tpot_data_hos.csv')

# ---------------------------------------------
# Train
# ---------------------------------------------
# The indexes used for cross validation
cvs_idxs = (data.cvs_hos_split == 'cvs')
hos_idxs = (data.cvs_hos_split == 'hos')

# Create matrices train
X_train = data[cvs_idxs][features].to_numpy()
y_train = data[cvs_idxs][labels].to_numpy()

# Create matrices test
X_test = data[cvs_idxs][features].to_numpy()
y_test = data[cvs_idxs][labels].to_numpy()

# ---------------------------------------------
# Search
# ---------------------------------------------
# Create genetic search
tpot = TPOTClassifier(generations=5, verbosity=2,
                      scoring='roc_auc', cv=2)

# Fit
tpot.fit(X_train, y_train)

# Score
score = tpot.score(X_test, y_test)

# Save
tpot.export('outcomes/main01/tpot_best_pipeline.py')