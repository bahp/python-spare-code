"""
TableOne
--------
"""
# Libraries
import pandas as pd

# Specific
from tableone import TableOne

# ------------------------
# Load data
# ------------------------
# Load data
data = pd.read_csv('./data/20210602-134857-completed/dataset.csv')

# ------------------------
# Create tableone
# ------------------------
# Columns
columns = ['age', 'gender', 'haematocrit_percent', 'plt']

# Categorical
categorical = ['gender']

# Groupby
groupby = ['cvs_hos_split']

#
mytable = TableOne(data, columns, categorical, groupby)

########################################################
# Show
#

mytable.tableone