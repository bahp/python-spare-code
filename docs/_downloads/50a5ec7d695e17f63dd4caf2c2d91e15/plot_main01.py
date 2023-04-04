"""
TableOne Basic Example
----------------------
"""
# Libraries
import pandas as pd

from tableone import TableOne

# Load data
url="https://raw.githubusercontent.com/tompollard/data/master/primary-biliary-cirrhosis/pbc.csv"
data=pd.read_csv(url)

# List of columns
columns = [
    'age', 'bili', 'albumin', 'ast',
    'platelet', 'protime', 'ascites',
    'hepato', 'spiders', 'edema', 'sex',
    'trt'
]

# Specify categorical columns
categorical = ['ascites','hepato','edema','sex','spiders','trt']

# Define groupy and not normal
groupby = 'trt'
nonnormal = ['bili']

# Create descriptive table
mytable = TableOne(data, columns, categorical,
                   groupby, nonnormal, pval=True)

# Show

#%%
# Convert to html
html = mytable.to_html()
html

#%%
# Lets tabulate for github.
print(mytable.tabulate(tablefmt="github"))

#%%
# Save as latex file
mytable.to_latex('mytable.tex')