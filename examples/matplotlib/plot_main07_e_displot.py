"""
07.e ``sns.displot``
============================================

Figure-level interface for drawing distribution plots onto a FacetGrid.

"""

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

sns.set_style(style="white")


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
path = Path('../../datasets/shap')
data = pd.read_csv(path / 'shap.csv')
data = data[data.features.isin(['C-Reactive Protein'])]

# Since the colorbar is discrete, needs to round so that
# the amount of bins is small and therefore visible. Would
# it be possible to define a continuous colormap?
data.feature_values = data.feature_values.round(1)

# Show
print(data.describe())

# Configuration
cmap_name = 'coolwarm' # colormap name


# .. note:: The function displot calls the histplot function. However,
#           the features allowed are count, frequency, probability or
#           proportion, percent and density. Thus, the median cannot
#           be computed.

# .. note:: The resulting colormap is discrete. Could it be continuous?

# Loop
for i, (name, df) in enumerate(data.groupby('features')):

    # Info
    print("%2d. Computing... %s" % (i, name))

    # Get colormap
    values = df.feature_values
    cmap, norm = scalar_colormap(values=values,
        cmap=cmap_name, vmin=values.min(),
        vmax=values.max())

    # Display displot
    sns.displot(data=df, x='timestep', y='shap_values',
        hue='feature_values', palette='coolwarm',
        hue_norm=(values.min(), values.max()),
        rug=False) # bins

    """
    # Display histplot
    plt.figure()
    sns.histplot(
        data=df, x='timestep', y='shap_values',
        discrete=(False, False),
        hue='feature_values', palette=cmap_name,
        hue_norm=(values.min(), values.max()),
        cbar=False, cbar_kws=dict(shrink=.75),
        #pthresh=.05, pmax=.9, bins=100
    )
    """

    # Format figure
    plt.suptitle(name)
    plt.tight_layout()
    plt.legend([], [], frameon=False)

    # Show only first N
    if int(i) > 2:
        break

# Show
plt.show()