"""
05c. Custom using swarmplot
=====================================

This script demonstrates how to build a custom SHAP visualization
for sequential data using ``seaborn.swarmplot``. This technique
creates a detailed, per-feature view that shows the distribution
of SHAP values at each timestep, offering a high-density
alternative to standard summary plots.

The script's workflow includes:

    - **Loading & Subsetting Data:** It loads a pre-computed, tidy
      DataFrame of SHAP values and subsets it to ensure the
      computationally intensive swarmplot runs efficiently.
    - **Per-Feature Visualization:** The script iterates through each
      feature, generating a separate swarmplot to clearly display
      its specific impact across the entire time sequence.
    - **Custom Value-Based Coloring:** It implements a custom coloring
      function to tint each point based on its original feature
      value, adding a color bar to provide the rich context found
      in native SHAP plots.
    - **Plot Customization:** It showcases how to fine-tune the plot's
      appearance, including normalizing the y-axis and managing legends
      for a polished final output.

This example is perfect for users who need to visualize the precise
distribution of feature impacts over time, though it is best suited
for smaller datasets where the swarmplot can arrange points without
significant overlap.
"""

# Libraries
import shap
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.cm

from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False


# ------------------------
# Methods
# ------------------------
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

def load_shap_file():
    """Load shap file.

    .. note: The timestep does not indicate time step but matrix
             index index. Since the matrix index for time steps
             started in negative t=-T and ended in t=0 the
             transformation should be taken into account.

    """
    from pathlib import Path
    # Load data
    path = Path('../../datasets/shap/')
    data = pd.read_csv(path / 'shap.csv')
    data = data.iloc[:, 1:]
    data = data.rename(columns={'timestep': 'indice'})
    data['timestep'] = data.indice - (data.indice.nunique() - 1)
    return data



# ------------------------------------------------------
#                       Main
# ------------------------------------------------------
# Configuration
cmap_name = 'coolwarm' # colormap name
norm_shap = True

# Load data
data = load_shap_file()

# Filter so that it is less computationally expensive
data = data[data['sample'] < 100]

# Show
if TERMINAL:
    print("\nShow:")
    print(data)

#%%
# Let's see how data looks like
data.head(10)


#%%
# Display using ``sns.swarmplot``
#
# .. warning:: This method seems to be quite slow.
#
# .. note:: y-axis has been 'normalized'
#

def add_colorbar(fig, cmap, norm):
    """"""
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb,
         cmap=cmap, norm=norm, orientation='vertical')


# Loop
for i, (name, df) in enumerate(data.groupby('features')):

    # Get colormap
    values = df.feature_values
    cmap, norm = scalar_palette(values=values,
        cmap=cmap_name, vmin=values.min(),
        vmax=values.max())

    # Display
    fig, ax = plt.subplots()
    ax = sns.swarmplot(x='timestep',
                       y='shap_values',
                       hue='feature_values',
                       palette=cmap,
                       data=df,
                       size=2,
                       ax=ax)

    # Format figure
    plt.title(name)
    plt.legend([], [], frameon=False)

    if norm_shap:
        plt.ylim(data.shap_values.min(),
                 data.shap_values.max())

    # Invert x axis (if no negative timesteps)
    #ax.invert_xaxis()

    # Create colormap (fix for old versions of mpl)
    cmap = matplotlib.cm.get_cmap(cmap_name)

    # Add colorbar
    add_colorbar(plt.gcf(), cmap, norm)

    # Show only first N
    if int(i) > 5:
        break

# Show
plt.show()
