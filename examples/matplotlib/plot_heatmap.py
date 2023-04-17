import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

# Load data
data = pd.read_csv('./data/shap.csv')
x = data.timestep
y = data.shap_values
z = data.feature_values

# Show
print(x.unique())

# Binned stats
statistic, x_edge, y_edge, binnumber = \
    stats.binned_statistic_2d(x=x, y=y, values=z,
        statistic='count', bins=[20, x.nunique()],
        expand_binnumbers=False)

sns.heatmap(statistic, annot=True, linewidth=.5, cmap='coolwarm')

plt.show()