"""
02. Basic example
-----------------

Basic usage of the tableone library.
"""
# Libraries
import pandas as pd

# Specific
from pathlib import Path
from tableone import TableOne


# ------------------------
# Load data
# ------------------------
# Load data
path = Path('../../datasets/dengue-htd-dataset')
data = pd.read_csv(path / 'dengue.csv')

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

########################################################
# Show (HTML)
#
# Html
html = mytable.to_html()

# show
print(html)
