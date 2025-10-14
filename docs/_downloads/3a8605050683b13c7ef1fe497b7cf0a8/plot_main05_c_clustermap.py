"""
05.c ``sns.clustermap`` multiple categories
-------------------------------------------

Plot a matrix dataset as a hierarchically-clustered heatmap.

.. note:: The hierarchical clustering has been deactivated.

"""
# Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.pyplot import gcf

#%%
# Let's load the dataset

# Load dataset
networks = sns.load_dataset("brain_networks",
    index_col=0, header=[0, 1, 2])

#%%
# Let's create the network colors

# Create network colors
network_labels = networks.columns.get_level_values("network")
network_pal = sns.cubehelix_palette(network_labels.unique().size,
    light=.9, dark=.1, reverse=True, start=1, rot=-2)
network_lut = dict(zip(map(str, network_labels.unique()), network_pal))

network_colors = \
    pd.Series(network_labels, index=networks.columns) \
        .map(network_lut)

#%%
# Let's create the node colors

# Create node colors
node_labels = networks.columns.get_level_values("node")
node_pal = sns.cubehelix_palette(node_labels.unique().size)
node_lut = dict(zip(map(str, node_labels.unique()), node_pal))

node_colors = \
    pd.Series(node_labels, index=networks.columns) \
        .map(node_lut)

#%%
# Let's combine them.

# Combine
network_node_colors = \
    pd.DataFrame(network_colors) \
        .join(pd.DataFrame(node_colors))

#%%
# Let's display the clustermap

# Display
g = sns.clustermap(networks.corr(),
    row_cluster=False, col_cluster=False, # turn off clusters
    row_colors = network_node_colors, # add colored labels
    col_colors = network_node_colors, # add colored labels
    linewidths=0, xticklabels=False, yticklabels=False,
    center=0, cmap="vlag", figsize=(7, 7))


# Add legend for networks
for label in network_labels.unique():
    g.ax_col_dendrogram.bar(0, 0,
        color=network_lut[label], label=label, linewidth=0)

l1 = g.ax_col_dendrogram.legend(title='Network',
    loc="center", ncol=5, bbox_to_anchor=(0.53, 0.9),
    bbox_transform=gcf().transFigure)

# Add legend for nodes
for label in node_labels.unique():
    g.ax_row_dendrogram.bar(0, 0,
        color=node_lut[label], label=label, linewidth=0)

l2 = g.ax_row_dendrogram.legend(title='Node',
    loc="center", ncol=1, bbox_to_anchor=(0.86, 0.9),
    bbox_transform=gcf().transFigure)

plt.show()