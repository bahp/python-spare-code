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

#
from pathlib import Path
from itertools import combinations
from matplotlib.colors import LogNorm, Normalize

# See https://matplotlib.org/devdocs/users/explain/customizing.html
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

"""
# Tuples
from collections import Counter

tuples = df.index.values
tuples = [tuple(sorted(t)) for t in tuples]
c = Counter(tuples)
"""

# print(c.most_common()[0])

def _check_ax_ay_equal(ax, ay):
    return ax==ay

def _check_ax_ay_greater(ax, ay):
    return  ax>ay

# Constants
figsize = (17, 4)

# Load data
#data = pd.read_csv('cri-urine.csv')
#data = pd.read_csv('./outputs/cri/20230524-171855/contingency.csv')
data = pd.read_csv('./outputs/cri/20230525-121300/contingency.csv')

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

    m.to_csv('%s'%str(i))

    # Add frequency
    top_n = df \
        .sort_values('samples', ascending=False) \
        .head(20).drop(columns='MIS') \
        .dropna(axis=1, how='all')

    # Draw
    fig, axs = plt.subplots(nrows=1, ncols=4,
        sharey=False, sharex=False, figsize=figsize,
        gridspec_kw={'width_ratios': [2, 3, 3, 3.5]})

    sns.heatmap(data=mis * 100, annot=False, linewidth=.5,
                cmap='coolwarm', vmin=-70, vmax=70, center=0,
                annot_kws={"size": 8}, square=True,
                ax=axs[2], xticklabels=True, yticklabels=True)

    sns.heatmap(data=freq, annot=False, linewidth=.5,
                cmap='Blues', norm=LogNorm(),
                annot_kws={"size": 8}, square=True,
                ax=axs[1], xticklabels=True, yticklabels=True)

    sns.heatmap(top_n,
                annot=False, linewidth=0.5,
                cmap='Blues', ax=axs[0], zorder=1,
                vmin=None, vmax=None, center=None, robust=True,
                square=False, xticklabels=True, yticklabels=True,
                cbar_kws={
                    'use_gridspec': True,
                    'location': 'right'
                }
    )

    # Possible cmaps
    # https://r02b.github.io/seaborn_palettes/
    # Diverging: coolwarm, RdBu_r, vlag
    # Others: bone, gray, pink, twilight
    cmap0 = 'coolwarm'
    cmap1 = sns.light_palette("seagreen", as_cmap=True)
    cmap2 = sns.color_palette("light:b", as_cmap=True)
    cmap3 = sns.color_palette("vlag", as_cmap=True)
    cmap4 = sns.diverging_palette(220, 20, as_cmap=True)
    cmap5 = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True) # no
    cmap6 = sns.color_palette("light:#5A9", as_cmap=True)
    cmap7 = sns.light_palette("#9dedcc", as_cmap=True)
    cmap8 = sns.color_palette("YlGn", as_cmap=True)

    # Display
    masku = np.triu(np.ones_like(m))
    maskl = np.tril(np.ones_like(m))
    sns.heatmap(data=m, cmap=cmap8, mask=masku, ax=axs[3],
                annot=False, linewidth=0.5, norm=LogNorm(),
                annot_kws={"size": 8}, square=True, vmin=0)
    sns.heatmap(data=m, cmap=cmap4, mask=maskl, ax=axs[3],
                annot=False, linewidth=0.5, vmin=-0.7, vmax=0.7,
                center=0, annot_kws={"size": 8}, square=True,
                xticklabels=True, yticklabels=True)

    # Configure axes
    axs[0].set_title('Contingency')
    axs[1].set_title('Number of samples')
    axs[2].set_title('Collateral Sensitivity Index')
    axs[3].set_title('Samples / Collateral Sensitivity')

    # Add colors to xticklabels

    abxs = pd.read_csv('../../datasets/susceptibility-nhs/susceptibility-v0.0.1/antimicrobials.csv')

    groups = dict(zip(abxs.antimicrobial_code, abxs.category))
    cmap = sns.color_palette("Spectral", abxs.category.nunique())
    colors = dict(zip(abxs.category, cmap))

    #colors = {
    #    'Aminoglycosides': 'blue',
    #    'Aminopenicillins': 'red',
    #    'Macrolides': 'green',
    #}

    for i in axs[3].get_xticklabels():
        try:
            t = i.get_text()
            x, y = i.get_position()
            c = colors.get(groups[t], 'k')
            i.set_color(c)

            print(i, x, y)
            #i.set_bbox(dict(facecolor='none', edgecolor='red'))

            from matplotlib.patches import Rectangle
            axs[3].add_patch(Rectangle((x-0.4, 23.0), 0.85, 0.5,
                edgecolor='k', facecolor=c, fill=True, lw=0.25),
                             zorder=6)

            """
            axs[3].annotate('', xy=(2000, 0),
                #xytext=(0, -15 - axs[3].xaxis.labelpad),
                xytext=(i.x, y)
                xycoords=('data', 'axes fraction'),
                textcoords='offset points',
                ha='center', va='top',
                bbox=dict(boxstyle='round', fc='none', ec='red'))
            """

        except Exception as e:
            print(i.get_text(), e)


    # Configure plot
    plt.suptitle(i)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)

    break

# Show
plt.show()