"""
01. Basic example
-----------------

Basic usage of the tableone library.

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

# Define groupby and not normal
groupby = 'trt'
nonnormal = ['bili']

# Create descriptive table
mytable = TableOne(data, columns, categorical,
                   groupby, nonnormal, pval=True)


#%%
mytable.tableone

#%%
# Convert to html
html = mytable.to_html()
html

#%%
# Lets tabulate for github.
print(mytable.tabulate(tablefmt="github"))

#%%
# Lets tabulate for latex.
print(mytable.tabulate(tablefmt="latex"))

#%%
# Lets tabulate for grid.
print(mytable.tabulate(tablefmt="grid"))

#%%
# Lets tabulate for fancy grid.
print(mytable.tabulate(tablefmt="fancy_grid"))

#%%
# Lets tabulate for markdown
print(mytable.tabulate(tablefmt="rst"))

#%%
# Save as latex file
mytable.to_latex('./outputs/main01-mytable.tex')

#%%
# Save as xls file
mytable.to_excel('./outputs/main01-mytable.xlsx')