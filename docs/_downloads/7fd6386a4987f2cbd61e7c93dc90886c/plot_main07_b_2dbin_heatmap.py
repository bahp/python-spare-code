"""
07.b ``stats.2dbin`` and ``sns.heatmap``
----------------------------------------

Use binned_statistic_2d and display using heatmap.

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats

# Load data
path = Path('../../datasets/shap/')
data = pd.read_csv(path / 'shap.csv')

# Get x, y and z
x = data.timestep
y = data.shap_values
z = data.feature_values

# Show
data[['timestep', 'shap_values', 'feature_values']]

# Binned stats
statistic, x_edge, y_edge, binnumber = \
    stats.binned_statistic_2d(x=x, y=y, values=z,
        statistic='count', bins=[20, x.nunique()],
        expand_binnumbers=False)

# Display
sns.heatmap(statistic, annot=True, linewidth=.5,
    cmap='coolwarm', annot_kws={"fontsize":6},
    square=False)

# Show
plt.show()