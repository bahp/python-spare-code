import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set_theme(style="dark")


def scalar_colormap(values, cmap, vmin, vmax):
    """This method creates a colormap based on values.

    Parameters
    ----------
    values : array-like
    The values to create the corresponding colors

    cmap : str
    The colormap

    vmin, vmax : float
    The minimum and maximum possible values

    Returns
    -------
    scalar colormap
    """
    # Create scalar mappable
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Get color map
    colormap = sns.color_palette([mapper.to_rgba(i) for i in values])
    # Return
    return colormap, norm

def scalar_palette(values, cmap, vmin, vmax):
    """This method creates a colorpalette based on values.

    Parameters
    ----------
    values : array-like
    The values to create the corresponding colors

    cmap : str
    The colormap

    vmin, vmax : float
    The minimum and maximum possible values

    Returns
    -------
    scalar colormap

    """
    # Create a matplotlib colormap from name
    # cmap = sns.light_palette(cmap, reverse=False, as_cmap=True)
    cmap = sns.color_palette(cmap, as_cmap=True)
    # Normalize to the range of possible values from df["c"]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # Create a color dictionary (value in c : color from colormap)
    colors = {}
    for cval in values:
        colors.update({cval: cmap(norm(cval))})
    # Return
    return colors, norm


# Load dataset
data = pd.read_csv('./data/shap.csv')
data = data[data.features.isin(['C-Reactive Protein'])]
data = data[data.timestep == 6]
#data.feature_values = data.feature_values * 100
data.feature_values = data.feature_values.round(1)
#data = data.head(1000)
#print(data)
print(data.describe())

# Configuration
cmap_name = 'coolwarm' # colormap name
norm_shap = True


import numpy as np
from scipy import stats


x = data.timestep
y = data.shap_values
z = data.feature_values

binx = np.array([5.5, 6.5])
biny = np.linspace(-1, 1, 10)

print(binx)
print(biny)

# Compute binned statistic (median)
r3 = stats.binned_statistic_2d(x=x, y=y, values=z,
    statistic='median', bins=[20, 20],
    expand_binnumbers=False)

print(r3.statistic)
print(r3.x_edge)
print(r3.y_edge)

plt.imshow(r3.statistic)
plt.colorbar()
plt.show()

import sys
sys.exit()

# Loop
for i, (name, df) in enumerate(data.groupby('features')):

    # Get colormap
    values = df.feature_values
    cmap, norm = scalar_colormap(values=values,
        cmap=cmap_name, vmin=values.min(),
        vmax=values.max())

    print(i)

    # Display
    """
    ax = sns.displot(data=df, x='timestep', y='shap_values',
        hue='feature_values', palette='coolwarm',
        hue_norm=(values.min(), values.max()),
        rug=False)
    """

    print(df)

    ax = sns.histplot(
        data=df, x='timestep', y='shap_values',
        discrete=(False, False),
        hue='feature_values', palette='coolwarm',
        hue_norm=(values.min(), values.max()),
        cbar=False, cbar_kws=dict(shrink=.75),
        #pthresh=.05, pmax=.9, bins=100
    )

    # Format figure
    plt.title(name)
    plt.legend([], [], frameon=False)

    #if norm_shap:
    #    plt.ylim(data.shap_values.min(),
    #             data.shap_values.max())

    # Invert x axis (if no negative timesteps)
    # ax.invert_xaxis()

    # Create colormap (fix for old versions of mpl)
    #cmap = matplotlib.cm.get_cmap(cmap_name)

    # Add colorbar
    #add_colorbar(plt.gcf(), cmap, norm)
    break
    # Show only first N
    if int(i) > 2:
        break

# Show
plt.show()


"""
values = data.feature_values
cmap, hnorm = scalar_colormap(values=values,
         cmap='coolwarm', vmin=values.min(),
         vmax=values.max())


# Display
sns.displot(
    data=data, x="timestep", y="shap_values", hue='feature_values',
    col="features", log_scale=(False, False), height=4, aspect=.7,
    col_wrap=2, palette=cmap, kind='hist'
)
"""
plt.show()