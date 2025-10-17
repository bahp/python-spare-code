"""
08. Energy forecast 2050 (sankey)
===================================

This script creates a Sankey diagram to visualize the projected
flow of energy for the year 2050. It begins by fetching the necessary
data from a JSON file hosted on GitHub.

Before plotting, the script processes the data to customize the
diagram's appearance, notably adjusting the colors and opacity
so that each flow (or link) inherits the color of its source node.
This enhances readability and makes the chart more intuitive.
Finally, it uses plotly.graph_objects to construct and display
the Sankey diagram, which effectively illustrates the distribution
of energy from various sources through to their final consumption
sectors.
"""
import plotly.graph_objects as go
import urllib.request, json

url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())

# override gray link colors with 'source' colors
opacity = 0.4
# change 'magenta' to its 'rgba' value to add opacity
data['data'][0]['node']['color'] = \
    ['rgba(255,0,255, 0.8)' if color == "magenta" else color
        for color in data['data'][0]['node']['color']]
data['data'][0]['link']['color'] = \
    [data['data'][0]['node']['color'][src].replace("0.8", str(opacity))
        for src in data['data'][0]['link']['source']]

print(data)

fig = go.Figure(data=[go.Sankey(
    valueformat = ".0f",
    valuesuffix = "TWh",
    # Define nodes
    node = dict(
      pad = 15,
      thickness = 15,
      line = dict(color = "black", width = 0.5),
      label =  data['data'][0]['node']['label'],
      color =  data['data'][0]['node']['color']
    ),
    # Add links
    link = dict(
      source =  data['data'][0]['link']['source'],
      target =  data['data'][0]['link']['target'],
      value =  data['data'][0]['link']['value'],
      label =  data['data'][0]['link']['label'],
      color =  data['data'][0]['link']['color']
))])

title = """Energy forecast for 2050<br>Source: Department of Energy & Climate 
Change, Tom Counsell ia <a href='https://bost.ocks.org/mike/sankey/'>Mike Bostock</a>","""

fig.update_layout(title_text=title, font_size=10)
#fig.show()

# Show
from plotly.io import show
show(fig)
