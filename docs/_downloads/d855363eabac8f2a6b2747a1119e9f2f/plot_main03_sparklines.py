"""
Main 03 - Plot Sparklines
=========================

This example shows to to plot sparklines style graphs.
"""
# -------------------
# Main
# -------------------
# https://chart-studio.plotly.com/~empet/13748/sparklines/#/code
# https://omnipotent.net/jquery.sparkline/#s-about
# https://chart-studio.plotly.com/create/?fid=Dreamshot:8025#/

# Libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pandas.tseries.offsets import DateOffset
from plotly.subplots import make_subplots
from plotly.io import show

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False

# Constants
colors = px.colors.sequential.Viridis_r

# Size
S = 100
N = 7

# Create data
x = np.arange(S)
y = np.random.randint(low=1, high=100, size=(S, N))

# Create DataFrame
data = pd.DataFrame(y)
data.columns = ['c%s'%i for i in data.columns]

# Create timedelta
data['timedelta'] = \
    pd.to_timedelta(data.index / 1, unit='D')

# Create datetimes (if needed)
today = pd.to_datetime('today').normalize()
data['dates'] = pd.to_datetime(today)
data['dates'] += pd.to_timedelta(data.index / 1, unit='D')

# Set dates as index
data = data.set_index('dates')

# Drop timedelta
data = data.drop(columns='timedelta')

# Show data
if TERMINAL:
    print("\nData:")
    print(data)
data



# ----------------
# Visualize
# ----------------
# Create layout
layout = {
  "font": {"family": "Georgia, serif"},
  #"title": "Sparklines",
  "width": 700,
  "height": 500,
  "margin": {"t": 80},
  "paper_bgcolor": 'rgba(0,0,0,0)',  # transparent
  "plot_bgcolor": 'rgba(0,0,0,0)',  # transparent
  "autosize": False,
  "hovermode": "closest",
  "showlegend": False,
}

# Create figure
fig = make_subplots(rows=N, cols=1,
                    subplot_titles=None)

# Add traces
for i, column in enumerate(data.columns):
    # Colors
    c = colors[i]
    x = data.index
    y = data[column]

    # Add trace
    fig.add_trace(go.Scatter(x=x, y=y,
        name=column.upper(),
        mode='lines', fill='tozeroy',
        line=dict(color=c, width=0.5),
        xaxis='x%s' % (i+1), yaxis='y%s' % (i+1)),
        row=i+1, col=1)

    # Update axes
    fig.update_yaxes(title_text=column, row=i+1, col=1)

    # Add to layout
    layout["xaxis%s" % (i+1)] = {
        "ticks": "",
        "anchor": 'y%s' % (i+1),
        "domain": [0.0, 1.0],
        "mirror": False,
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False
    }
    layout["yaxis%s" % (i+1)] = {
      "ticks": "",
      "anchor": 'x%s' % (i+1),
      #"domain": [0.08416666666666667, 0.15833333333333333],
      "mirror": False,
      "showgrid": False,
      "showline": False,
      "zeroline": False,
      "showticklabels": False
    }

# Update layout
fig.update_layout(layout)

# Show
show(fig)