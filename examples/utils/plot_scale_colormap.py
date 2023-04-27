"""
Show colormap for various scalings
----------------------------------

This dummy piece of code is to demonstrate visually that
the transformation using the MinMaxScaler or StandardScaler
does not affect the colormap.

These transformations compute a 'mapping' but do not
alter the values and/or their distribution. See the
formulas below:

MinMaxScaler = std * (x_max - x_min) / x_max
where std = (x - x_min) / (x_max - x_min)

StandardScaler = (x - x_mean) / x_std

"""

# Library
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# See https://matplotlib.org/devdocs/users/explain/customizing.html
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

# Create data
linear = np.linspace(0, 10, 100)
rayleigh = np.random.rayleigh(3, 100)
original = np.concatenate((linear, rayleigh)).reshape(-1, 1)
#original = rayleigh.reshape(-1, 1)

# Create scalers
mmx = MinMaxScaler().fit(original)
std = StandardScaler().fit(original)

# Display
fig, axs = plt.subplots(nrows=1, ncols=4,
    sharey=True, sharex=False, figsize=(8, 7))

for i, (name, data) in enumerate(
        [('original', original),
         ('MinMax', mmx.transform(original)),
         ('StdScaler', std.transform(original)),
         ('log', np.log(original))]):

    sns.heatmap(data.reshape(10, -1), annot=False,
        linewidth=0.5, cmap='coolwarm', ax=axs[i], zorder=1,
        cbar_kws={
            # 'label': 'value [unit]',
            'use_gridspec': True,
            'location': 'right'
        }
    )
    axs[i].set_title(name)


fig, axs = plt.subplots(nrows=3, ncols=1,
    sharey=True, sharex=False, figsize=(8, 7))

for i, (name, data) in enumerate(
        [('original', original),
         ('MinMax', mmx.transform(original)),
         ('StdScaler', std.transform(original))]):
    #sns.displot(data=data.reshape(1, -1).tolist(), ax=axs[i],
    #bins=100)
    #kind="kde")
    #sns.kdeplot(data=data.reshape(1, -1).tolist(), ax=axs)
    sns.histplot(data=data.reshape(1, -1).tolist(), ax=axs[i], kde=True)
    axs[i].set_title(name)


# Show
plt.show()
