"""
Main 31 - Sample shap.csv boxplot
---------------------------------

The aim is to visualise all the features for all the timesteps
to quickly see which shap values are higher and therefore
influence more in the result.

"""

# Libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px

from plotly.io import show
from plotly.colors import n_colors
from plotly.express.colors import sample_colorscale

# See https://matplotlib.org/devdocs/users/explain/customizing.html
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handleheight'] = 1
mpl.rcParams['legend.loc'] = 'upper left'

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False

# Features
features = [
    'Ward Lactate',
    #'Ward Glucose',
    #'Ward sO2',
    #'White blood cell count, blood',
    'Platelets',
    'Haemoglobin',
    'Mean cell volume, blood',
    'Haematocrit',
    #'Mean cell haemoglobin conc, blood',
    #'Mean cell haemoglobin level, blood',
    #'Red blood cell count, blood',
    #'Red blood cell distribution width',
    #'Creatinine',
    #'Urea level, blood',
    #'Potassium',
    #'Sodium',
    'Neutrophils',
    'Chloride',
    'Lymphocytes',
    'Monocytes',
    'Eosinophils',
    'C-Reactive Protein',
    'Albumin',
    #'Alkaline Phosphatase',
    #'Glucose POCT Strip Blood',
    'Total Protein',
    'Globulin',
    'Alanine Transaminase',
    'Bilirubin',
    'Prothrombin time',
    'Fibrinogen (clauss)',
    'Procalcitonin',
    'Ferritin',
    'D-Dimer',
    'sex',
    'age'
]

# Load data
data = pd.read_csv('../../datasets/shap/shap.csv')

# Filter
data = data[data.features.isin(features)]

if TERMINAL:
    print("\nData:")
    print(data)
data

# Number of colors
N = data.features.nunique()

# see https://plotly.com/python/builtin-colorscales/#discrete-color-sequences
# see https://plotly.github.io/plotly.py-docs/generated/plotly.express.box.html

# generate an array of rainbow colors by fixing the saturation and lightness of the
# HSL representation of colour and marching around the hue. Plotly accepts any CSS
# color format, see e.g. http://www.w3schools.com/cssref/css_colors_legal.asp.
c0 = ['hsl('+str(h)+',50%'+',50%)'
    for h in np.linspace(0, 360, N)]

# More colors
x = np.linspace(0, 1, N)
c1 = sample_colorscale('viridis', list(x))
c2 = sample_colorscale('RdBu', list(x))
c3 = sample_colorscale('Jet', list(x))
c4 = sample_colorscale('Agsunset', list(x))

# Boxplot
#plt.figure(figsize=(12, 4))
fig = px.box(data, x='timestep', y='shap_values',
             color='features', color_discrete_sequence=c4,
             points='outliers')

# .. note:: If using widescreen, commenting the legend section
#           will automatically generate a vertical legend with
#           scrolling if needed. For display purposes in the
#           docs we have included the legend on top.

# Update layout
fig.update_layout(
    #margin={
    #    'l': 0,
    #    'r': 0,
    #    'b': 0,
    #    't': 0,
    #    'pad': 4
    #},
    legend=dict(
        orientation="h",
        entrywidth=128,
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    paper_bgcolor='rgba(0,0,0,0)',  # transparent
    plot_bgcolor='rgba(0,0,0,0)'  # transparent
)

# Update xaxis
fig.update_xaxes(
    mirror=False,
    ticks='outside',
    showline=False,
    linecolor='black',
    gridcolor='lightgrey'
)

# Update yaxis
fig.update_yaxes(
    mirror=False,
    ticks='outside',
    showline=False,
    linecolor='black',
    gridcolor='lightgrey'
)

# Show
show(fig)
