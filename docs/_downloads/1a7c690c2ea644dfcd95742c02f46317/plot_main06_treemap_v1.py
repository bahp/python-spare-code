"""
Main 06 - Plot Treemap with coffee
----------------------------------

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

# Load dataframe
df = pd.read_csv(URL)

# Show
if TERMINAL:
    print("\nDF:")
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