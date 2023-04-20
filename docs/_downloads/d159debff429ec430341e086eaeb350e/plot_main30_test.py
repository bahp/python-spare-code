"""
Main 07 - 2dbin with shap.csv
-----------------------------

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

# Load data
data = pd.read_csv('./data/shap.csv')

# Filter
data = data[data.features.isin([
    'Ward Lactate'
    'Alanine Transaminase',
    'Albumin',
    'Alkaline Phosphatase',
    'Bilirubin',
    'C-Reactive Protein',
    'Chloride',
    'Creatinine',
    'Platelets',
    'Haemoglobin',
    'Mean cell volume, blood',
    'Haematocrit'
])]

# Show
print(data.head(10))



# .. todo:: Change flier size, cmap, ...

ax = sns.boxenplot(data, x='timestep', y='shap_values',
    hue='features', saturation=0.5, showfliers=False)
sns.despine(ax=ax)

plt.figure()
ax = sns.violinplot(data, x='timestep', y='shap_values',
    hue='features', saturation=0.5, showfliers=False)
sns.despine(ax=ax)

plt.figure(figsize=(12, 4))
ax = sns.boxplot(data, x='timestep', y='shap_values',
    hue='features', saturation=0.5, showfliers=False)
sns.despine(ax=ax)
plt.setp(ax.get_legend().get_texts(), fontsize='7')

plt.show()
