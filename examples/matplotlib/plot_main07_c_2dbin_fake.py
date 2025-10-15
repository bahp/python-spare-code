"""
07.c ``stats.2dbin`` with fake data
============================================

Use binned_statistic_2d and display using heatmap.

"""

# Libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from matplotlib.colors import LogNorm

def data_shap():
    data = pd.read_csv('../../datasets/shap/shap.csv')
    return data.timestep, \
           data.shap_values, \
           data.feature_values, \
           data

def data_manual():
    """"""
    # Create random values
    x = np.array([1, 1, 1, 1, 2, 2, 2, 3])
    y = np.array([1, 1, 2, 2, 3, 3, 4, 1])
    z = np.array([1, 1, 5, 6, 7, 8, 7, 7])
    return x, y, z, None

# Create data
x, y, z, data = data_manual()

print(x)
print(y)
print(z)

"""
# With pandas
v = z
vals, bins = np.histogram(v)
a = pd.Series(v).groupby(pd.cut(v, bins)).median()
print("\nPandas:")
print(bins)
print(vals)
print(a)
"""







vmin = z.min()
vmax = z.max()



# Compute binned statistic (median)
binx = np.linspace(0, 3, 4) + 0.5
biny = np.linspace(0, 4, 5) + 0.5
binx = np.linspace(0, 6, 7) + 0.5
biny = np.linspace(y.min(), y.max(), 100)
r1 = stats.binned_statistic_2d(x=y, y=x, values=z,
    statistic='count', bins=[biny, binx],
    expand_binnumbers=False)

r2 = stats.binned_statistic_2d(x=y, y=x, values=z,
    statistic='median', bins=[biny, binx],
    expand_binnumbers=False)

# Compute centres
x_center = (r1.x_edge[:-1] + r1.x_edge[1:]) / 2
y_center = (r1.y_edge[:-1] + r1.y_edge[1:]) / 2

# Show
print("\nScipy:")
print(binx)
print(biny)
print(r1.statistic)
print(x_center)
print(y_center)

# Convert the computed matrix to an stacked dataframe?

flip1 = np.flip(r1.statistic, 0)
flip2 = np.flip(r2.statistic, 0)
#flip1 = r1.statistic
#flip2 = r2.statistic



# Display
fig, axs = plt.subplots(nrows=1, ncols=2,
    sharey=False, sharex=False, figsize=(8, 7))

sns.heatmap(flip1, annot=False, linewidth=0.5,
    xticklabels=y_center.astype(int),
    yticklabels=x_center.round(2)[::-1], # Because of flip
    cmap='Blues', ax=axs[0],
    norm=LogNorm())

sns.heatmap(flip2, annot=False, linewidth=0.5,
    xticklabels=y_center.astype(int),
    yticklabels=x_center.round(2)[::-1], # Because of flip
    cmap='coolwarm', ax=axs[1], zorder=1,
    vmin=None, vmax=None, center=None, robust=True)

# If robust=True and vmin or vmax are absent, the colormap range
# is computed with robust quantiles instead of the extreme values.
"""
sns.violinplot(x=x, y=y,
    saturation=0.5, fliersize=0.1, linewidth=0.5,
    color='green', ax=axs[2], zorder=3,
    width=0.5)
"""


# Configure ax0
axs[0].set_title('count')
axs[0].set_xlabel('timestep')
axs[0].set_ylabel('shap')
axs[0].locator_params(axis='y', nbins=10)
#axs[0].set_aspect('equal', 'box'

# Configure ax1
axs[1].set_title('median')
axs[1].set_xlabel('timestep')
axs[1].set_ylabel('shap')
axs[1].locator_params(axis='y', nbins=10)
#axs[1].set_aspect('equal', 'box')
#axs[1].invert_yaxis()

# Generic
plt.suptitle('C-Reactive Protein')

"""
# Set axes manually
#plt.set_xticks()
#plt.setp(axs[1].get_yticklabels()[::1], visible=False)
#plt.setp(axs[1].get_yticklabels()[::5], visible=True)
from matplotlib import ticker
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(1.00))
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
"""

plt.tight_layout()
plt.show()


"""
import sys
sys.exit()
# Compute bin statistic (count and median)

# Plot
plt.figure()
sns.violinplot(data=data, x="timestep", y="shap_values", inner="box")
plt.figure()
plt.tight_layout()
sns.violinplot(data=data, x="timestep", y="feature_values", inner="box")
plt.figure()
sns.histplot(data=data, x="timestep", shrink=.8)


# Plot hist
f1 = plt.hist2d(data.timestep, data.feature_values, bins=30, cmap='Reds')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.title('Counts (square bin)')
plt.show()
"""