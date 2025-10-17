"""
02. Summary statistic for Dengue data
=========================================================

This script demonstrates a fundamental application of the tableone
Python library for creating a clinical summary table. Using a dengue
patient dataset, it showcases the process of summarizing patient
characteristics. The script loads the data, selects key demographic
and clinical variables (like age, gender, and platelet count), and
defines a grouping variable to compare these characteristics across
different patient subgroups. Finally, it generates and displays the
formatted summary table.
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

print(data)
print(data.columns)
# ------------------------
# Create tableone
# ------------------------
# Columns
columns = ['age', 'gender', 'haematocrit_percent', 'plt']

# Categorical
categorical = ['gender']

# Groupby
groupby = 'cvs_hos_split'

#
mytable = TableOne(data, columns=columns,
    categorical=categorical, groupby=groupby)

########################################################
# Lets see the table
#

mytable.tableone

########################################################
# Lets show the raw HTML
#
# Html
html = mytable.to_html()

# show
print(html)
