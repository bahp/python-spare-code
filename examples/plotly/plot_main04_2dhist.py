"""
04. 2D Contour & marginal histograms
====================================

This script demonstrates how to create a combined visualization using Plotly.
It displays the relationship between two variables using a 2D density contour
plot overlaid with a scatter plot of the individual data points. To show the
distribution of each variable independently, marginal histograms are added to
the top and right axes.

.. note:: Example from Plotly: https://plotly.com/python/2d-histogram-contour/

"""

import numpy as np
import plotly.graph_objects as go

from plotly.io import show

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False

# ----------------
# Create data
# ----------------
t = np.linspace(-1, 1.2, 2000)
x = (t**3) + (0.3 * np.random.randn(2000))
y = (t**6) + (0.3 * np.random.randn(2000))

# ----------------
# Create figure
# ----------------
fig = go.Figure()
fig.add_trace(go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'Blues', # 'Jet'
        contours = dict(
            showlabels = True,
            labelfont = dict(
                family = 'Raleway',
                color ='white'
            )
        ),
        hoverlabel = dict(
            bgcolor = 'white',
            bordercolor = 'black',
            font = dict(
                family = 'Raleway',
                color = 'black'
            )
        ),
        reversescale = True,
        xaxis = 'x',
        yaxis = 'y'
    ))
fig.add_trace(go.Scatter(
        x = x,
        y = y,
        xaxis = 'x',
        yaxis = 'y',
        mode = 'markers',
        marker = dict(
            color = 'rgba(0,0,0,0.3)',
            size = 3
        )
    ))
fig.add_trace(go.Histogram(
        y = y,
        xaxis = 'x2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))
fig.add_trace(go.Histogram(
        x = x,
        yaxis = 'y2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))

fig.update_layout(
    autosize = False,
    xaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    yaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    xaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    yaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    height = 600,
    width = 600,
    bargap = 0,
    hovermode = 'closest',
    showlegend = False
)

# Show
show(fig)