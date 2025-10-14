"""
05. ``sns.clustermap`` basic sample
-------------------------------------

Plot a matrix dataset as a hierarchically-clustered heatmap.

.. note:: The hierarchical clustering has been deactivated.

"""

"""
# Display 0
iris = sns.load_dataset("iris")
species = iris.pop("species")
lut = dict(zip(species.unique(), "rbg"))
row_colors = species.map(lut)
"""

# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
N = 20
C = sns.color_palette("Spectral", n_colors=5, as_cmap=False)

# Create data
data = np.random.randint(low=0, high=10, size=(N, N))
colors = [np.random.choice(['r', 'g', 'b']) for i in range(N)]
series = pd.Series({'A%s'%i:'Group %s'%c for i,c in enumerate(colors)})

# Create DataFrame
df = pd.DataFrame(data,
    index=['A%s'%i for i in range(N)],
    columns=['A%s'%i for i in range(N)]
)

# Show
print(df)
print(colors)
print(series)

# Display 1
# ---------
# Create colors dictionary
col_colors = dict(zip(df.columns, colors))

# Show
g = sns.clustermap(df,
    figsize=(5,5),
    row_cluster=False, col_cluster=False,
    #row_colors=col_colors,
    col_colors=pd.Series(col_colors),
    linewidths=0,
    xticklabels=False, yticklabels=False,
    center=0, cmap="vlag"
)

# Display 2
# ---------
# Create colors dictionary

col_colors = series.map({
    'Group g': 'g',
    'Group b': 'b',
    'Group r': 'r'
})

# Show
g = sns.clustermap(df,
    figsize=(5,5),
    row_cluster=False, col_cluster=False,
    #row_colors=col_colors,
    col_colors=pd.Series(col_colors),
    linewidths=0,
    xticklabels=False, yticklabels=False,
    center=0, cmap="vlag"
)

# Show
plt.show()