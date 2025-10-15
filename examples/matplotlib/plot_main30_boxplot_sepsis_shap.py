"""
11. Visualizing SHAP Value Distributions Across Timesteps
=========================================================

This script analyzes the feature importance from a time-series model
by visualizing pre-computed SHAP values.

This script loads SHAP (SHapley Additive exPlanations) values from a CSV
file to explore feature importance in a temporal context. After filtering
for a predefined list of medical features, it generates three distinct
Seaborn plots: a boxenplot, a violin plot, and a standard boxplot. Each
plot visualizes the distribution of SHAP values for every feature across
multiple timesteps. The main objective is to identify which features have
the most significant impact on the model's predictions (indicated by higher
SHAP values) and to observe how this influence evolves over time.


.. note:: See plotly example, were interaction with data is possible!

"""

# Libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats
from matplotlib.colors import LogNorm

sns.set_theme(style="white")

# See https://matplotlib.org/devdocs/users/explain/customizing.html
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handleheight'] = 1
mpl.rcParams['legend.loc'] = 'upper left'

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

# Show
print(data.head(10))


# .. todo:: Change flier size, cmap, ...


def configure_ax(ax):
    sns.despine(ax=ax)
    lg = ax.legend(loc='upper center',
                   bbox_to_anchor=(0.05, 1.15, 0.9, 0.1),
                   borderaxespad=2, ncol=5, mode='expand')
    plt.tight_layout()

# Boxenplot
plt.figure(figsize=(12, 4))
ax = sns.boxenplot(data, x='timestep', y='shap_values',
    hue='features', saturation=0.5, showfliers=False)
configure_ax(ax)

# Violinplot
plt.figure(figsize=(12, 4))
ax = sns.violinplot(data, x='timestep', y='shap_values',
    hue='features', saturation=0.5)
configure_ax(ax)

# Boxplot
plt.figure(figsize=(12, 4))
ax = sns.boxplot(data, x='timestep', y='shap_values',
    hue='features', saturation=0.5, showfliers=False,
    whis=1.0)
configure_ax(ax)


# Show
plt.show()
