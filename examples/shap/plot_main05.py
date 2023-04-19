"""
Shap - Main 05
==============ss
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
    #cmap = sns.light_palette(cmap, reverse=False, as_cmap=True)
    cmap = sns.color_palette(cmap, as_cmap=True)
    # Normalize to the range of possible values from df["c"]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # Create a color dictionary (value in c : color from colormap)
    colors = {}
    for cval in values:
        colors.update({cval : cmap(norm(cval))})
    # Return
    return colors, norm


def create_random_shap(samples, timesteps, features):
    """Create random LSTM data.

    .. note: No need to create the 3D matrix and then reshape to
             2D. It would be possible to create directly the 2D
             matrix.

    Parameters
    ----------
    samples: int
        The number of observations
    timesteps: int
        The number of time steps
    features: int
        The number of features

    Returns
    -------
    Stacked matrix with the data.

    """
    # .. note: Either perform a pre-processing step such as
    #          normalization or generate the features within
    #          the appropriate interval.
    # Create dataset
    x = np.random.randint(low=0, high=100,
        size=(samples, timesteps, features))
    y = np.random.randint(low=0, high=2, size=samples).astype(float)
    i = np.vstack(np.dstack(np.indices((samples, timesteps))))

    # Create DataFrame
    df = pd.DataFrame(
        data=np.hstack((i, x.reshape((-1, features)))),
        columns=['sample', 'timestep'] + \
                ['f%s'%j for j in range(features)]
    )

    df_stack = df.set_index(['sample', 'timestep']).stack()
    df_stack = df_stack
    df_stack.name = 'shap_values'
    df_stack = df_stack.to_frame()
    df_stack.index.names = ['sample', 'timestep', 'features']
    df_stack = df_stack.reset_index()

    df_stack['feature_values'] = np.random.randint(
        low=0, high=100, size=df_stack.shape[0])

    return df_stack


def load_shap_file():
    data = pd.read_csv('./data/shap.csv')
    data = data.iloc[: , 1:]
    #data.timestep = data.timestep - (data.timestep.nunique() - 1)
    return data

#################################################################
# Lets generate and/or load the shap values.

# .. note: The right format to use for plotting depends
#          on the library we use. The data structure is
#          good when using seaborn
# Load data
data = create_random_shap(10, 6, 4)
#data = load_shap_file()
#data = data[data['sample'] < 100]

shap_values = pd.pivot_table(data,
        values='shap_values',
        index=['sample', 'timestep'],
        columns=['features'])

feature_values = pd.pivot_table(data,
        values='feature_values',
        index=['sample', 'timestep'],
        columns=['features'])

# Show
if TERMINAL:
    print("\nShow:")
    print(data)
    print(shap_values)
    print(feature_values)

#%%
# Let's see how data looks like
data.head(10)

#%%
# Let's see how shap_values looks like
shap_values.iloc[:10, :5]

#%%
# Let's see how feature_values looks like
feature_values.iloc[:10, :5]


########################################################################
# Display using ``shap.summary_plot``
# -----------------------------------------------
#
# The first option is to use the ``shap`` library to plot the results.

# Let's define/extract some useful variables.
N = 4                                                       # max loops filter
TIMESTEPS = len(shap_values.index.unique(level='timestep')) # number of timesteps
SAMPLES = len(shap_values.index.unique(level='sample'))     # number of samples

shap_min = data.shap_values.min()
shap_max = data.shap_values.max()

#%%
# Now, let's display the shap values for all features in each timestep.

# For each timestep (visualise all features)
for i, step in enumerate(range(TIMESTEPS)[:N]):
    # Show
    #print('%2d. %s' % (i, step))

    # .. note: First option (commented) is only necessary if we work
    #          with a numpy array. However, since we are using a DataFrame
    #          with the timestep, we can index by that index level.
    # Compute indices
    #indice = np.arange(SAMPLES)*TIMESTEPS + step
    indice = shap_values.index.get_level_values('timestep') == i

    # Create auxiliary matrices
    shap_aux = shap_values.iloc[indice]
    feat_aux = feature_values.iloc[indice]

    # Display
    plt.figure()
    plt.title("Timestep: %s" % i)
    shap.summary_plot(shap_aux.to_numpy(), feat_aux, show=False)
    plt.xlim(shap_min, shap_max)

#%%
# Now, let's display the shap values for all timesteps of each feature.

# For each feature (visualise all time-steps)
for i, f in enumerate(shap_values.columns[:N]):
    # Show
    #print('%2d. %s' % (i, f))

    # Create auxiliary matrices (select feature and reshape)
    shap_aux = shap_values.iloc[:, i] \
        .to_numpy().reshape(-1, TIMESTEPS)
    feat_aux = feature_values.iloc[:, i] \
        .to_numpy().reshape(-1, TIMESTEPS)
    feat_aux = pd.DataFrame(feat_aux,
        columns=['timestep %s'%j for j in range(TIMESTEPS)]
    )

    # Show
    plt.figure()
    plt.title("Feature: %s" % f)
    shap.summary_plot(shap_aux, feat_aux, sort=False, show=False)
    plt.xlim(shap_min, shap_max)

#%%
# .. note:: If y-axis represents timesteps the ``sort`` parameter
#           in the ``summary_plot`` function is set to False.

########################################################################
# Display using ``sns.stripplot``
# -------------------------------
#
# .. warning:: This method seems to be quite slow.
#
# Let's display the shap values for each feature and all time steps.
# In contrast to the previous example, the timesteps are now displayed
# on the x-axis and the y-axis contains the shap values.


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
    cmap, norm = scalar_palette(values=values, cmap='coolwarm',
        vmin=values.min(), vmax=values.max())

    print(df)

    # Display
    fig, ax = plt.subplots()
    ax = sns.stripplot(x='timestep',
                       y='shap_values',
                       hue='feature_values',
                       palette=cmap,
                       data=df,
                       ax=ax)

    # Needed for older matplotlib versions
    cmap = matplotlib.cm.get_cmap('coolwarm')

    # Configure axes
    plt.title(name)
    plt.legend([], [], frameon=False)
    ax.invert_xaxis()
    add_colorbar(plt.gcf(), cmap, norm)

    # End
    if int(i) > N:
        break

# Show
plt.show()


########################################################################
# Display using ``sns.swarmplot``
# -------------------------------
#
# .. note: If the number of samples is too high, the points overlap
#          and are ignored by the ``swarmplot`` library. In such scenario
#          it is better to use ``stripplot``.
#
#
# Let's display the shap values for each timestep.

# Loop
for i, (name, df) in enumerate(data.groupby('features')):

    # Get colormap
    values = df.feature_values
    cmap, norm = scalar_palette(values=values, cmap='coolwarm',
        vmin=values.min(), vmax=values.max())

    # Display
    fig, ax = plt.subplots()
    ax = sns.swarmplot(x='timestep',
                       y='shap_values',
                       hue='feature_values',
                       palette=cmap,
                       data=df,
                       size=2,
                       ax=ax)

    # Needed for older matplotlib versions
    cmap = matplotlib.cm.get_cmap('coolwarm')

    # Configure axes
    plt.title(name)
    plt.legend([], [], frameon=False)
    ax.invert_xaxis()
    add_colorbar(plt.gcf(), cmap, norm)

    # End
    if int(i) > N:
        break

# Show
plt.show()







"""
sns.set_theme(style="ticks")

# Create a dataset with many short random walks
rs = np.random.RandomState(4)
pos = rs.randint(-1, 2, (20, 5)).cumsum(axis=1)
pos -= pos[:, 0, np.newaxis]
step = np.tile(range(5), 20)
walk = np.repeat(range(20), 5)
df = pd.DataFrame(np.c_[pos.flat, step, walk],
                  columns=["position", "step", "walk"])
# Initialize a grid of plots with an Axes for each walk
#grid = sns.FacetGrid(df_stack, col="walk", hue="f", palette="tab20c",
#                     col_wrap=4, height=1.5)

grid = sns.FacetGrid(df_stack, hue="f",
    palette="tab20c", height=1.5)

# Draw a horizontal line to show the starting point
grid.refline(y=0, linestyle=":")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "t", "value", marker="o")

# Adjust the tick positions and labels
grid.set(xticks=np.arange(5), yticks=[-3, 3],
         xlim=(-.5, 4.5), ylim=(-3.5, 3.5))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)

"""


#plt.show()

##########################################################################
# Display using ``sns.FacetGrid``
# -------------------------------
#

#g = sns.FacetGrid(df_stack, col="f", hue='original')
#g.map(sns.swarmplot, "t", "value", alpha=.7)
#g.add_legend()


##########################################################################
# Display using ``shap.beeswarm``
# -------------------------------
#

# REF: https://github.com/slundberg/shap/blob/master/shap/plots/_beeswarm.py
#
# .. note: It needs a kernel explainer, and while it works with
#          common kernels (plot_main07.py) it does not work with
#          the DeepKernel for some reason (mask related).