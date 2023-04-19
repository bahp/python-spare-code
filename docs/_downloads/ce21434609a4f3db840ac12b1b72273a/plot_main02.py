"""
TableOne - Main02
=================
"""
# Libraries
import pandas as pd

# Specific
from tableone import TableOne


# ------------------------
# Load data
# ------------------------
# Load data
data = pd.read_csv('./data/dengue.csv')

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
