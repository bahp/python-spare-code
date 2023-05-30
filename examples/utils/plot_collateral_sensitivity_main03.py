"""
Collateral Sensitivity Index (Display)
--------------------------------------

Since the computation of the Collateral Sensitivity Index is quite
computationally expensive, the results are saved into a .csv file
so that can be easily loaded and displayed. This script shows a
very basic graph of such information.

"""
# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# Specific libraries
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, Normalize

# See https://matplotlib.org/devdocs/users/explain/customizing.html
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

# --------------------------------------
# Helper methods
# --------------------------------------
def _check_ax_ay_equal(ax, ay):
    return ax==ay

def _check_ax_ay_greater(ax, ay):
    return  ax>ay

# Constants
figsize = (8, 5)

# Load data
#data = pd.read_csv('cri-urine.csv')
#data = pd.read_csv('./outputs/cri/20230524-171855/contingency.csv')
data = pd.read_csv('./outputs/cri/20230525-121300/contingency.csv')
abxs = pd.read_csv('../../datasets/susceptibility-nhs/susceptibility-v0.0.1/antimicrobials.csv')

print(data)

# Format data
data = data.set_index(['specimen', 'o', 'ax', 'ay'])
data.RR = data.RR.fillna(0).astype(int)
data.RS = data.RS.fillna(0).astype(int)
data.SR = data.SR.fillna(0).astype(int)
data.SS = data.SS.fillna(0).astype(int)
data['samples'] = data.RR + data.RS + data.SR + data.SS

#data['samples'] = data.iloc[:, :4].sum(axis=1)

def filter_top_pairs(df, n=5):
    """Filter top n (Specimen, Organism) pairs."""
    # Find top
    top = df.groupby(level=[0, 1]) \
        .samples.sum() \
        .sort_values(ascending=False) \
        .head(n)

    # Filter
    idx = pd.IndexSlice
    a = top.index.get_level_values(0).unique()
    b = top.index.get_level_values(1).unique()

    # Return
    return df.loc[idx[a, b, :, :]]

# Filter
data = filter_top_pairs(data, n=2)
data = data[data.samples > 500]

# Show
print("\n")
print("Number of samples: %s" % data.samples.sum())
print("Number of pairs: %s" % data.shape[0])
print("Data:")
print(data)


#####################################################################
# Lets load the antimicrobial data and create color mapping variables

# Create dictionary to map category to color
labels = abxs.category
palette = sns.color_palette('colorblind', labels.nunique())
palette = sns.cubehelix_palette(labels.nunique(),
    light=.9, dark=.1, reverse=True, start=1, rot=-2)
lookup = dict(zip(labels.unique(), palette))

# Create dictionary to map code to category
code2cat = dict(zip(abxs.antimicrobial_code, abxs.category))


# Loop
for i, df in data.groupby(level=[0, 1]):

    # Drop level
    df = df.droplevel(level=[0, 1])

    # Check possible issues.
    ax = df.index.get_level_values(0)
    ay = df.index.get_level_values(1)
    idx1 = _check_ax_ay_equal(ax, ay)
    idx2 = _check_ax_ay_greater(ax, ay)

    # Show
    print("%25s. ax==ay => %5s | ax>ay => %5s" % \
          (i, idx1.sum(), idx2.sum()))

    # Re-index to have square matrix
    abxs = set(ax) | set(ay)
    index = pd.MultiIndex.from_product([abxs, abxs])

    # Reformat MIS
    mis = df['MIS'] \
        .reindex(index, fill_value=np.nan) \
        .unstack()

    # Reformat samples
    freq = df['samples'] \
        .reindex(index, fill_value=0) \
        .unstack()

    # Combine in square matrix
    m1 = mis.copy(deep=True).to_numpy()
    m2 = freq.to_numpy()
    il1 = np.tril_indices(mis.shape[1])
    m1[il1] = m2.T[il1]
    m = pd.DataFrame(m1,
        index=mis.index, columns=mis.columns)

    # ------------------------------------------
    # Display heatmaps
    # ------------------------------------------
    # Create color maps
    cmapu = sns.color_palette("YlGn", as_cmap=True)
    cmapl = sns.diverging_palette(220, 20, as_cmap=True)

    # Masks
    masku = np.triu(np.ones_like(m))
    maskl = np.tril(np.ones_like(m))

    # Draw figure
    fig, axs = plt.subplots(nrows=1, ncols=1,
        sharey=False, sharex=False, figsize=figsize)

    # Create own colorbar axes
    # Params are [left, bottom, width, height]
    cbar_ax1 = fig.add_axes([0.76, 0.5, 0.03, 0.38])
    cbar_ax2 = fig.add_axes([0.90, 0.5, 0.03, 0.38])

    # Display
    r1 = sns.heatmap(data=m, cmap=cmapu, mask=masku, ax=axs,
                     annot=False, linewidth=0.5, norm=LogNorm(),
                     annot_kws={"size": 8}, square=True, vmin=0,
                     cbar_ax=cbar_ax2,
                     cbar_kws={'label': 'Number of isolates'})

    r2 = sns.heatmap(data=m, cmap=cmapl, mask=maskl, ax=axs,
                     annot=False, linewidth=0.5, vmin=-0.7, vmax=0.7,
                     center=0, annot_kws={"size": 8}, square=True,
                     xticklabels=True, yticklabels=True,
                     cbar_ax=cbar_ax1,
                     cbar_kws={'label': 'Collateral Resistance Index'})


    # ------------------------------------------
    # Add category rectangular patches
    # ------------------------------------------
    # Create colors
    colors = m.columns.to_series().map(code2cat).map(lookup)

    # Create patches for categories
    category_patches = []
    for lbl in axs.get_xticklabels():
        try:
            x, y = lbl.get_position()
            c = colors.to_dict().get(lbl.get_text(), 'k')
            # i.set_color(c) # for testing

            # Add patch.
            category_patches.append(
                Rectangle((x - 0.35, y - 0.5), 0.8, 0.3, edgecolor='k',
                          facecolor=c, fill=True, lw=0.25, alpha=0.5, zorder=1000,
                          transform=axs.transData
                          )
            )
        except Exception as e:
            print(lbl.get_text(), e)

    # Add category rectangles
    fig.patches.extend(category_patches)


    # ------------------------------------------
    # Add category legend
    # ------------------------------------------
    # Unique categories
    unique_categories = m.columns \
        .to_series().map(code2cat).unique()

    # Create legend elements
    legend_elements = [
        Patch(facecolor=lookup.get(k, 'k'), edgecolor='k',
              fill=True, lw=0.25, alpha=0.5, label=k)
        for k in unique_categories
    ]

    # Add legend
    axs.legend(handles=legend_elements, loc='lower left',
               ncol=1, bbox_to_anchor=(1.1, 0.00), fontsize=8,
               fancybox=False, shadow=False)

    # Configure plot
    plt.suptitle('%s - %s' % (i[0], i[1]))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)



# Show
plt.show()