"""
MPL2PLY SHAP summary
====================

.. note:: In the latest commit of plotly packages/python/plotly/plotly/matplotlylib/mpltools.py line 368,
          it still calls is_frame_like() function. There is already an issue tracking this. You may need
          choose to downgrade Matplotlib if you still want to use mpl_to_plotly() function.

"""
# Generic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Xgboost
from xgboost import XGBClassifier

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False


# ----------------------------------------
# Load data
# ----------------------------------------
# Seed
seed = 0

# Load dataset
bunch = load_iris()
bunch = load_breast_cancer()

# Features
features = list(bunch['feature_names'])

# Create DataFrame
data = pd.DataFrame(data=np.c_[bunch['data'],
    bunch['target']], columns=features + ['target'])

# Create X, y
X = data[bunch['feature_names']]
y = data['target']

# Filter
X = X.iloc[:500, :3]
y = y.iloc[:500]

# Split dataset
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state=seed)


# ----------------------------------------
# Classifiers
# ----------------------------------------
# Define some classifiers
gnb = GaussianNB()
llr = LogisticRegression()
dtc = DecisionTreeClassifier(random_state=seed)
rfc = RandomForestClassifier(random_state=seed)
xgb = XGBClassifier(
    min_child_weight=0.005,
    eta= 0.05, gamma= 0.2,
    max_depth= 4,
    n_estimators= 100)

# Select one
clf = xgb

# Fit
clf.fit(X_train, y_train)

# ----------------------------------------
# Compute shap values
# ----------------------------------------
# Import
import shap

# Get generic explainer
explainer = shap.Explainer(clf, X_train)

# Show kernel type
print("\nKernel type: %s" % type(explainer))

# Get shap values
shap_values = explainer(X)

# Show shap values
print("Shap values:")
print(shap_values)
print(shap_values.values.shape)
print(shap_values.base_values.shape)
print(shap_values.data.shape)

# Get matplotlib figure
plot_summary = shap.summary_plot( \
    explainer.shap_values(X_train),
    X_train, cmap='viridis',
    show=False)

# Show
#plt.show()

"""
# Convert to plotly
import plotly.tools as tls
import plotly.graph_objs as go

# Get current figure and convert
fig = tls.mpl_to_plotly(plt.gcf())

# Format
# Update layout
fig.update_layout(
    #xaxis_title='False Positive Rate',
    #yaxis_title='True Positive Rate',
    #yaxis=dict(scaleanchor="x", scaleratio=1),
    #xaxis=dict(constrain='domain'),
    width=700, height=350,
    #legend=dict(
    #    x=1.0, y=0.0,  # x=1, y=1.02
    #    orientation="v",
    #    font=dict(
    #       size=12,
    #        color='black'),
    #    yanchor="bottom",
    #    xanchor="right",
    #),
    font=dict(
        size=15,
        #family="Times New Roman",
        #color="black",
    ),
    title=dict(
        font=dict(
        #    family="Times New Roman",
        #    color="black"
        )
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=[0, 1, 2],
        ticktext=features[:-1],
        tickfont=dict(size=15)
    ),
    xaxis=dict(
        tickfont=dict(size=15)),
    #margin={
    #    'l': 0,
    #    'r': 0,
    #    'b': 0,
    #    't': 0,
    #    'pad': 4
    #},
    paper_bgcolor='rgba(0,0,0,0)',  # transparent
    plot_bgcolor='rgba(0,0,0,0)',   # transparent
    template='simple_white'
)

# Update scatter
fig.update_traces(marker={'size': 10})

# Add vertical lin
fig.add_vline(x=0.0, line_width=2,
    line_dash="dash", line_color="black") # green

# .. note:: Would it be possible to get the values of
#           cmin, cmax and the tick vals from the shap
#           values? Ideally we do not want to hardcode
#           them.

# Add colorbar
colorbar_trace = go.Scatter(
    x=[None], y=[None], mode='markers',
    marker=dict(
        colorscale='viridis',
        showscale=True,
        cmin=-5,
        cmax=5,
        colorbar=dict(thickness=20,
            tickvals=[-5, 5],
            ticktext=['Low', 'High'],
            outlinewidth=0)),
    hoverinfo='none'
)
fig['layout']['showlegend'] = False
fig.add_trace(colorbar_trace)

# Show
#fig.show()
fig
"""