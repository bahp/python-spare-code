"""
06. Coffee flavours (treemap)
====================================

This example uses the pandas and Plotly libraries to create
and display an interactive treemap visualization of coffee
flavors.

"""
# %%
# Lets load the dataset.

# Libraries
import plotly.graph_objects as go
import pandas as pd

from plotly.io import show

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False


# Define URL
URL = 'https://raw.githubusercontent.com/plotly/datasets/96c0bd/sunburst-coffee-flavors-complete.csv'

# Load DataFrame
df = pd.read_csv(URL)

# Show
if TERMINAL:
    print("\nDataFrame:")
    print(df)
df

# %%
#

# Display
fig = go.Figure(go.Treemap(
    ids=df.ids,
    labels=df.labels,
    parents=df.parents,
    pathbar_textfont_size=15,
    #maxdepth=3,
    root_color="lightgrey"
))

# Update
fig.update_layout(
    uniformtext=dict(minsize=10, mode='hide'),
    margin=dict(t=50, l=25, r=25, b=25)
)

# Show
show(fig)