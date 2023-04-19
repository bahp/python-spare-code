"""
Main 11 - Using updatemenus
===========================

This example show how to include buttons in the
plotly graph by using update menus. Work should
be put on control their placement and how to
avoid them to be missaligned when resizing the
window!

.. warning:: It is not finished!

"""

import plotly
import plotly.graph_objs as go

from plotly.io import show

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False


def get_color_set(color_set_id):
    """Get color set."""
    if color_set_id == 1:
        marker_color = ['red', 'green', 'blue']
    elif color_set_id == 2:
        marker_color = ['black', 'blue', 'red']
    return [{'marker.color': [marker_color]}];


# Define trace
trace = go.Scatter(
    x=[0,1,1],
    y=[1,0,1],
    marker=dict(color=['green','black','red']),
    mode='markers'
)

# Define updatemenus
updatemenus=list([
    dict(
        buttons=list([
            dict(label = 'Color Set 1',
                 method = 'update',
                 args=get_color_set(1)
            ),
            dict(label = 'Color Set 2',
                 method = 'update',
                 args=get_color_set(2)
            ),
        ]),
        direction = 'left',
        pad = {'r': 10, 't': 10},
        showactive = True,
        type = 'buttons',
        x = 0.1,
        xanchor = 'left',
        y = 1.1,
        yanchor = 'top'
    )
])

# Define layout
layout = go.Layout(
    title='Scatter Color Switcher',
    updatemenus = updatemenus
)

# Create Figure
fig = go.Figure(data=[trace], layout=layout)

# ----------------------------
# Save
# ----------------------------
# Libraries
import time
from pathlib import Path

# Define pipeline path
path = Path('./objects') / 'plot_main11_updatemenus'
filename = '%s.html' % time.strftime("%Y%m%d-%H%M%S")

# Create folder (if it does not exist)
path.mkdir(parents=True, exist_ok=True)

# Save
fig.write_html("%s/%s" % (path, filename))

# Show
show(fig)