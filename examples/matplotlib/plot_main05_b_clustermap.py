"""
05.b ``sns.clustermap`` with ``network.csv``
============================================

This script demonstrates an advanced use of ``seaborn.clustermap``
to visualize a correlation matrix from the "brain_networks"
dataset. 🧠 Its main purpose is to show how to add categorical
color annotations to rows and columns. It features a clever
workaround to create a proper legend for these annotations by
plotting invisible bars, resulting in a polished and informative
heatmap.

.. note:: The hierarchical clustering has been deactivated.

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
networks = sns.load_dataset("brain_networks",
    index_col=0, header=[0, 1, 2])

# Create variables
network_labels = networks.columns.get_level_values("network")
network_pal = sns.cubehelix_palette(
    network_labels.unique().size,
    light=.9, dark=.1, reverse=True,
    start=1, rot=-2)
network_lut = dict(zip(map(str, network_labels.unique()), network_pal))
network_colors = pd.Series(network_labels).map(network_lut)

# The side colors are drawn with a heatmap, which matplotlib thinks
# of as quantitative data and thus there's not a straightforward way
# to get a legend directly from it. Instead of that, we'll add an
# invisible barplot with the right colors and labels, then add a
# legend for that.
g = sns.clustermap(networks.corr(),
    row_cluster=False, col_cluster=False,                 # turn-off clustering
    row_colors=network_colors, col_colors=network_colors, # add class labels
    linewidths=0, xticklabels=False, yticklabels=False)   # improve visual wih many rows

for label in network_labels.unique():
    g.ax_col_dendrogram.bar(0, 0,
        color=network_lut[label],
        label=label, linewidth=0)

# Move the colorbar to the empty space.
g.ax_col_dendrogram.legend(loc="center", ncol=6)
g.cax.set_position([.15, .2, .03, .45])

# Show
plt.show()