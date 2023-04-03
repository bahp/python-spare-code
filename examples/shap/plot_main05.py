# Libraries
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    # Gete color map
    colormap = sns.color_palette([mapper.to_rgba(i) for i in values])
    # Return
    return colormap


def create_random_shap(samples, timesteps, features):
    """Create random LSTM data.

    .. note: No need to create the 3D matrix and then reshape to
             2D. It would be possible to create directly the 2D
             matrix.

    Parameters
    ----------

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
    data = pd.read_csv('./shap.csv')
    data = data.iloc[: , 1:]
    return data


# -----------------------------------------------------
# Main
# -----------------------------------------------------
# Load data
#data = create_random_shap(100, 5, 10)
data = load_shap_file()

shap_values = pd.pivot_table(data,
        values='shap_values',
        index=['sample', 'timestep'],
        columns=['features'])

feature_values = pd.pivot_table(data,
        values='feature_values',
        index=['sample', 'timestep'],
        columns=['features'])

# Show
print("\nShow:")
print(data)
print(shap_values)
print(feature_values)


# Method 0: Using shap library trick
# ----------------------------------
# The number of timesteps
TIMESTEPS = len(shap_values.index.unique(level='timestep'))
SAMPLES = len(shap_values.index.unique(level='sample'))

# For each timestep (visualise all features)
for i, step in enumerate(range(TIMESTEPS)):
    # Show
    print('%2d. %s' % (i, step))

    # .. note: First is only necessary if we work with a numpy
    #          array. However, since we are using a DataFrame
    #          with the timestep, we can index by that index
    #          level.
    # Compute indices
    #indice = np.arange(SAMPLES)*TIMESTEPS + step
    indice = shap_values.index.get_level_values('timestep') == i

    # Create auxiliary matrices
    shap_aux = shap_values.iloc[indice]
    feat_aux = feature_values.iloc[indice]

    # Display
    #shap.summary_plot(shap_aux.to_numpy(), feat_aux)


# For each feature (visualise all timesteps)
for i, f in enumerate(shap_values.columns[:2]):
    # Show
    print('%2d. %s' % (i, f))

    # Create auxiliary matrices (select feature and reshape)
    shap_aux = shap_values.iloc[:, i] \
        .to_numpy().reshape(-1, TIMESTEPS)
    feat_aux = feature_values.iloc[:, i] \
        .to_numpy().reshape(-1, TIMESTEPS)
    feat_aux = pd.DataFrame(feat_aux,
        columns=['timestep %s'%j for j in range(TIMESTEPS)]
    )

    # Show
    #shap.summary_plot(shap_aux, feat_aux, sort=False)


# Method 1:
# --------
for i, df in data.groupby('features'):

    f = sns.stripplot(x='timestep',
                      y='shap_values',
                      hue='feature_values',
                      palette='viridis', 
                      data=data)
    plt.title(i)
    plt.legend([], [], frameon=False)
    f.invert_xaxis()

# Show
plt.show()

import sys
sys.exit()

for i, df in df_stack.groupby('f'):


    col = 'value'

    colormap = scalar_colormap(df[col],
        cmap='RdYlBu', vmin=df[col].min(),
        vmax=df[col].max())

    plt.figure()
    f = sns.stripplot(x='t', y='value', hue='original',
        palette='viridis', data=df)
    plt.title(i)
    plt.legend([], [], frameon=False)
    f.invert_xaxis()

    """
    ax = sns.scatterplot(data=tips,
        x="sex", y="total_bill", hue="total_bill", palette="viridis", legend=False)
    pts = ax.collections[0]
    pts.set_offsets(pts.get_offsets() + np.c_[np.random.uniform(-.1, .1, len(tips)), np.zeros(len(tips))])
    ax.margins(x=.5)
    ax.autoscale_view()
    """



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


plt.show()

# Method 3: Facet Grid
# --------------------
#g = sns.FacetGrid(df_stack, col="f", hue='original')
#g.map(sns.swarmplot, "t", "value", alpha=.7)
#g.add_legend()

# Method 4: Using beeswarm
# ------------------------
# REF: https://github.com/slundberg/shap/blob/master/shap/plots/_beeswarm.py
#
# .. note: It needs a kernel explainer, and while it works with
#          common kernels (plot_main07.py) it does not work with
#          the DeepKernel for some reason (mask related).