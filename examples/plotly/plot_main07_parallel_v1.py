"""
07a. Manufacturing features (parallel)
=======================================

This Python script uses the pandas and Plotly libraries to create
an interactive parallel coordinates plot. This type of plot is excellent
for visualizing and exploring relationships in datasets with many variables
(high-dimensional data). Each vertical line represents a different variable
(a column from the dataset), and each colored line that snakes across the
plot represents a single data point (a row from the dataset).

"""

import plotly.graph_objects as go
import pandas as pd

from plotly.io import show

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False


# Load
df = pd.read_csv("https://raw.githubusercontent.com/bcdunbar/datasets/master/parcoords_data.csv")

# Visualize
if TERMINAL:
    print("\nData:")
    print(df)
df

# Show
fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['colorVal'],
                   colorscale = 'Electric',
                   showscale = True,
                   cmin = -4000,
                   cmax = -100),
        dimensions = list([
            dict(range = [32000,227900],
                 constraintrange = [100000,150000],
                 label = "Block Height", values = df['blockHeight']),
            dict(range = [0,700000],
                 label = 'Block Width', values = df['blockWidth']),
            dict(tickvals = [0,0.5,1,2,3],
                 ticktext = ['A','AB','B','Y','Z'],
                 label = 'Cyclinder Material', values = df['cycMaterial']),
            dict(range = [-1,4],
                 tickvals = [0,1,2,3],
                 label = 'Block Material', values = df['blockMaterial']),
            dict(range = [134,3154],
                 visible = True,
                 label = 'Total Weight', values = df['totalWeight']),
            dict(range = [9,19984],
                 label = 'Assembly Penalty Wt', values = df['assemblyPW']),
            dict(range = [49000,568000],
                 label = 'Height st Width', values = df['HstW'])])
    )
)

# Show
show(fig)