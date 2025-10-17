"""
07.b ``stats.2dbin`` and ``sns.heatmap``
============================================

This script offers a streamlined approach for visualizing 2D data
density by combining ``scipy`` for aggregation and ``seaborn`` for
plotting. 📊 It demonstrates a powerful and common data science
pattern for handling large, scattered datasets.

The workflow involves:

    - Binning Data: It uses ``scipy.stats.binned_statistic_2d``
      to group scattered 2D points into a grid and count the
      number of occurrences in each cell.
    - Seaborn Visualization: The resulting 2D count matrix is
      then directly plotted using ``seaborn.heatmap``, providing a
      quick and effective way to see data concentration.

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