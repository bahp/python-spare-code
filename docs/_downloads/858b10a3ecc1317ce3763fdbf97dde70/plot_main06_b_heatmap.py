"""
06.b ``sns.heatmap`` for CRI ``v2``
-------------------------------------------

Plot rectangular data as a color-encoded matrix.

The generates a heatmap visualization for a dataset related to collateral
sensitivity. It uses the Seaborn library to plot the rectangular data as
a color-encoded matrix. The code loads the data from a CSV file, creates
mappings for categories and colors, and then plots the heatmap using the
loaded data and color maps. It also includes annotations, colorbar axes,
category patches, legend elements, and formatting options to enhance the
visualization.

"""
# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

# See https://matplotlib.org/devdocs/users/explain/customizing.html
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False

############################################################
# Lets load the data and create color mapping variables

# Load data
path = Path('../../datasets/collateral-sensitivity/sample')
data = pd.read_csv(path / 'matrix.csv', index_col=0)
abxs = pd.read_csv(path / 'categories.csv', index_col=0)

# Create dictionary to map category to color
labels = abxs.category
palette = sns.color_palette('colorblind', labels.nunique())
lookup = dict(zip(labels.unique(), palette))

# Create dictionary to map code to category
code2cat = dict(zip(abxs.antimicrobial_code, abxs.category))

# Create colors
colors = data.columns.to_series().map(code2cat).map(lookup)


#%%
# Lets see the data
if TERMINAL:
    print("\nData:")
    print(data)
data.iloc[:7,:7]

#%%
# Lets see the antimicrobials
if TERMINAL:
    print("\nAntimicrobials:")
    print(abxs)
abxs


############################################################
# Lets create some variables.

# Create color maps
cmapu = sns.color_palette("YlGn", as_cmap=True)
cmapl = sns.diverging_palette(220, 20, as_cmap=True)

# Create triangular matrices
masku = np.triu(np.ones_like(data))
maskl = np.tril(np.ones_like(data))


############################################################
# Let's display a heatmap
#

# Draw (heatmap)
fig, axs = plt.subplots(nrows=1, ncols=1,
    sharey=False, sharex=False, figsize=(8, 5)
)

# Display
r1 = sns.heatmap(data=data, cmap=cmapu, mask=masku, ax=axs,
            annot=False, linewidth=0.5, norm=LogNorm(),
            annot_kws={"size": 8}, square=True, vmin=0,
            cbar_kws={'label': 'Number of isolates'})

r2 = sns.heatmap(data=data, cmap=cmapl, mask=maskl, ax=axs,
            annot=False, linewidth=0.5, vmin=-0.7, vmax=0.7,
            center=0, annot_kws={"size": 8}, square=True,
            xticklabels=True, yticklabels=True,
            cbar_kws={'label': 'Collateral Resistance Index'})

# Create patches for categories
category_patches = []
for i in axs.get_xticklabels():
    try:
        x, y = i.get_position()
        c = colors.to_dict().get(i.get_text(), 'k')
        #i.set_color(c) # for testing

        # Add patch.
        category_patches.append(
            Rectangle((x-0.35, y-0.5), 0.8, 0.3, edgecolor='k',
                facecolor=c, fill=True, lw=0.25, alpha=0.5, zorder=1000,
                transform=axs.transData
            )
        )
    except Exception as e:
        print(i.get_text(), e)

# Add category rectangles
fig.patches.extend(category_patches)

# Create legend elements
legend_elements = [
    Patch(facecolor=v, edgecolor='k',
        fill=True, lw=0.25, alpha=0.5, label=k)
            for k, v in lookup.items()
]

# Add legend
axs.legend(handles=legend_elements, loc='upper center',
    ncol=3, bbox_to_anchor=(0.5, -0.25), fontsize=8,
    fancybox=False, shadow=False)

# Format
plt.suptitle('URINE - Escherichia Coli')
#plt.subplots_adjust(right=0.2)
plt.tight_layout()

# Show
plt.show()