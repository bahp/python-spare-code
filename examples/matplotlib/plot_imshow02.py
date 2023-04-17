import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

# Load data
data = pd.read_csv('./data/shap.csv')
x = data.timestep
y = data.shap_values
z = data.feature_values

# Binned stats
statistic, x_edge, y_edge, binnumber = \
    stats.binned_statistic_2d(x=x, y=y, values=z,
        statistic='count', bins=[5, 10],
        expand_binnumbers=False)

print(statistic)

# Plot
plt.subplot(111)
plt.imshow(statistic)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.7, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.tight_layout()

# Show
plt.show()