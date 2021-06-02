"""
Plotly - ROC
===================

layout = Layout(
    xaxis=XAxis(
       range=[-150, 150],
       showgrid=True,
       zeroline=True,
       showline=True,
       gridcolor='#bdbdbd',
       gridwidth=2,
       zerolinecolor='#969696',
       zerolinewidth=4,
       linecolor='#636363',
       linewidth=6
    ),
    yaxis=YAxis(
        range=[-150,150],
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=2,
        zerolinecolor='#969696',
        zerolinewidth=4,
        linecolor='#636363',
        linewidth=6
   ),
   height=600,
   width=600,
)


"""




# Libraries
import plotly.graph_objects as go
from plotly.graph_objects import Layout
import plotly.express as px
import numpy as np


# -----------------------------------------
# Config
# -----------------------------------------
# Colors
colors = px.colors.qualitative.Plotly
colors = px.colors.sequential.Plasma_r
colors = px.colors.sequential.Viridis_r
#colors = px.colors.sequential.Blues # Issue only 9!

print(colors)

# -----------------------------------------
# Data
# -----------------------------------------
# Create some data
fpr = np.arange(10)/10
tpr = np.arange(10)/10

# Data
data = {
    'split1': np.vstack((fpr + 0.1, tpr)).T,
    'split2': np.vstack((fpr + 0.2, tpr)).T,
    'split3': np.vstack((fpr + 0.2, tpr)).T,
    'split4': np.vstack((fpr + 0.2, tpr)).T,
    'split5': np.vstack((fpr + 0.2, tpr)).T,
    'split6': np.vstack((fpr + 0.2, tpr)).T,
    'split7': np.vstack((fpr + 0.2, tpr)).T,
    'split8': np.vstack((fpr + 0.2, tpr)).T,
    'split9': np.vstack((fpr + 0.2, tpr)).T,
    'split10': np.vstack((fpr + 0.2, tpr)).T
}


# -------------------------------------
# Visualize
# -------------------------------------
# Create figure
fig = go.Figure()

# Add diagonal line
fig.add_shape(type='line', x0=0, x1=1, y0=0, y1=1,
    line=dict(dash='dash', color='gray', width=1),
)

# Plot each split
for i, (name, array) in enumerate(data.items()):
    # Name of split
    name = f"{name}" # (AUC={10:.2f})"
    # Add trace
    fig.add_trace(go.Scatter(x=array[:, 0],
                             y=array[:, 1],
                             name=name,
                             mode='lines+markers',
                             line=dict(color=colors[i],
                                       width=0.5)))

# Update layout
fig.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=350, height=350,
    legend=dict(
        x=1.0, y=0.0,  # x=1, y=1.02
        orientation="v",
        font={'size': 12},
        yanchor="bottom",
        xanchor="right",
    ),
    margin={
        'l': 0,
        'r': 0,
        'b': 0,
        't': 0,
        'pad': 4
    },
    paper_bgcolor='rgba(0,0,0,0)',  # transparent
    plot_bgcolor='rgba(0,0,0,0)'  # transparent
)

# Update xaxes
fig.update_xaxes(showgrid=True,
                 gridwidth=1,
                 nticks=10,
                 range=[0, 1],
                 gridcolor='lightgray')

# Update yaxes
fig.update_yaxes(showgrid=True,
                 gridwidth=1,
                 range=[0, 1],
                 nticks=10,
                 gridcolor='lightgray')

# Show
#fig.show()
fig