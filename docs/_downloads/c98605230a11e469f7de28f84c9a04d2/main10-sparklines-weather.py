"""
Plotly - Sparklines
===================
"""
# -------------------
# Main
# -------------------
# https://chart-studio.plotly.com/~empet/13748/sparklines/#/code
# https://omnipotent.net/jquery.sparkline/#s-about
# https://chart-studio.plotly.com/create/?fid=Dreamshot:8025#/

# Libraries
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pandas.tseries.offsets import DateOffset
from plotly.subplots import make_subplots

# Constants
colors = px.colors.sequential.Viridis_r
colors += px.colors.sequential.Viridis_r

# -------------------
# Load data
# -------------------
# Define folder
folder = './data/weather-hmc-dataset/'

# Load data
data = pd.concat([pd.read_csv(f) \
    for f in glob.glob('%s/*.csv' % folder)])

# Drop duplicates
data = data.drop_duplicates()
data['dates'] = pd.to_datetime(data['Date time'])
data = data.set_index('dates')
data = data.drop(columns=['Name',
                          'Date time',
                          'Conditions'])

# Show data
print(data)
print(data.shape)


# ----------------
# Visualize
# ----------------
# Create layout
layout = {
  "font": {"family": "Georgia, serif"},
  "title": "Weather Ho Chi Minh (missing units!)",
  #"width": 500,
  #"height": 500,
  #"margin": {"t": 80},
  "paper_bgcolor": 'rgba(0,0,0,0)',  # transparent
  "plot_bgcolor": 'rgba(0,0,0,0)',  # transparent
  #"autosize": False,
  "hovermode": "closest",
  "showlegend": False,
}

# Create figure
fig = make_subplots(rows=data.shape[1], cols=1,
                    shared_xaxes=True,
                    subplot_titles=[t for t in data.columns])

# Add traces
for i, column in enumerate(data.columns):
    # Colors
    c = colors[i]
    x = data.index
    y = data[column]

    # Add trace
    fig.add_trace(go.Scatter(x=x, y=y,
        name=column,
        mode='lines', fill='tozeroy',
        line=dict(color=c, width=0.5)),
        row=i+1, col=1)

    # Update axes
    fig.update_yaxes(title_text='', row=i+1, col=1)

# Update layout
fig.update_layout(layout)

# Show
fig.show()
fig