"""
Plot Distributions
======================

This example shows how to plot the probability distributions
for each of the components of the confusion matrix. Note that
data is created artificially. If necessary, it is possible
to limit the axis to the range [0, 1] to keep only those true
probability values.
"""

# Libraries
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Specific
from plotly.graph_objects import Layout

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False


# -----------------------------------------
# Helper method
# -----------------------------------------
# This method is implemented in pySML.
def _tp_fp_tn_fn_distributions(y, y_pred, y_prob):
    """This function returns probabilities for each of the confusion
    matrix elements (tp, tn, fp, fn).

    Parameters
    ----------
    y : array-like
      The real categories

    y_pred : array-like
      The predicted categories

    y_prob: array-like
      The predict probabilities

    Returns
    -------
    tp_probs, tn_probs, fp_probs, fn_probs
    """
    # Tags.
    tp_idx = (y_pred == 1) & (y == 1)
    tn_idx = (y_pred == 0) & (y == 0)
    fp_idx = (y_pred == 1) & (y == 0)
    fn_idx = (y_pred == 0) & (y == 1)
    # Show information.
    tp_probs = y_prob[tp_idx]
    tn_probs = y_prob[tn_idx]
    fp_probs = y_prob[fp_idx]
    fn_probs = y_prob[fn_idx]
    # Return
    return tp_probs, tn_probs, fp_probs, fn_probs

# -----------------------------------------
# Config
# -----------------------------------------
# Colors
colors = px.colors.qualitative.Plotly
colors = px.colors.sequential.Plasma_r
colors = px.colors.sequential.Viridis_r

# -----------------------------------------
# Data
# -----------------------------------------
# Create data
data = pd.DataFrame()
data['y_true'] = np.random.randint(2, size=100)
data['y_pred'] = np.random.randint(2, size=100)
data['y_prob'] = np.random.normal(loc=0, scale=1, size=100)

# Get distributions
tp_probs, tn_probs, fp_probs, fn_probs = \
    _tp_fp_tn_fn_distributions(data.y_true,
                               data.y_pred,
                               data.y_prob)

# Visualize
if TERMINAL:
    print("\nData:")
    print(data)


# -------------------------------------
# Visualize
# -------------------------------------
# Import subplots
from plotly.subplots import make_subplots

# Create figure
fig = make_subplots(rows=2, cols=2)
# subplot_titles=('TP', 'TN', 'FP', 'FN'))

#  Add traces
fig.add_trace(go.Violin(x=tn_probs, line_width=1,
    name='tn', line_color='black', fillcolor=colors[2],
    opacity=0.5, meanline_visible=True, box_visible=True), row=1, col=1)
fig.add_trace(go.Violin(x=fp_probs, line_width=1,
    name='fp', line_color='black', fillcolor=colors[4],
    opacity=0.5, meanline_visible=True, box_visible=True), row=1, col=2)
fig.add_trace(go.Violin(x=fn_probs, line_width=1,
    name='fn', line_color='black', fillcolor=colors[6],
    opacity=0.5, meanline_visible=True, box_visible=True), row=2, col=1)
fig.add_trace(go.Violin(x=tp_probs, line_width=1,
    name='tp', line_color='black', fillcolor=colors[0],
    opacity=0.5, meanline_visible=True, box_visible=True), row=2, col=2)

# Update layout
fig.update_layout(
    width=700, height=350,
    #xaxis_title='False Positive Rate',
    #yaxis_title='True Positive Rate',
    #yaxis=dict(scaleanchor="x", scaleratio=1),
    #xaxis=dict(constrain='domain'),
    #legend=dict(
    #    x=1.0, y=0.0,  # x=1, y=1.02
    #    orientation="v",
    #    font={'size': 12},
    #    yanchor="bottom",
    #    xanchor="right",
    #),
    margin={
        'l': 0,
        'r': 0,
        'b': 0,
        't': 0,
        'pad': 0
    },
    paper_bgcolor='rgba(0,0,0,0)',  # transparent
    plot_bgcolor='rgba(0,0,0,0)'  # transparent
)

# Update axes
#fig.update_xaxes(visible=True, range=[0.0, 0.5], row=1, col=1)
#fig.update_xaxes(visible=True, range=[0.5, 1.0], row=1, col=2)
#fig.update_xaxes(visible=True, range=[0.0, 0.5], row=2, col=1)
#fig.update_xaxes(visible=True, range=[0.5, 1.0], row=2, col=2)
fig.update_yaxes(visible=True)

# Show
if TERMINAL:
    fig.show()
fig