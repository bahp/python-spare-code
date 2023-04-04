"""
Shap - Main 05 - Summary plot
=============================
"""

# Libraries
import shap
import pandas as pd

import matplotlib.pyplot as plt


try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False


# ------------------------
# Methods
# ------------------------
def load_shap_file():
    """Load shap file.

    .. note: The timestep does not indicate time step but matrix
             index index. Since the matrix index for time steps
             started in negative t=-T and ended in t=0 the
             transformation should be taken into account.

    """
    data = pd.read_csv('./data/shap.csv')
    data = data.iloc[:, 1:]
    data = data.rename(columns={'timestep': 'indice'})
    data['timestep'] = data.indice - (data.indice.nunique() - 1)
    return data


# -----------------------------------------------------
#                       Main
# -----------------------------------------------------
# Load data
# data = create_random_shap(10, 6, 4)
data = load_shap_file()
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

# %%
# Let's see how data looks like
data.head(10)

# %%
# Let's see how shap_values looks like
shap_values.iloc[:10, :5]

# %%
# Let's see how feature_values looks like
feature_values.iloc[:10, :5]

########################################################################
# Display using ``shap.summary_plot``
# -----------------------------------------------
#
# The first option is to use the ``shap`` library to plot the results.

# Let's define/extract some useful variables.
N = 10  # max loops filter
TIMESTEPS = len(shap_values.index.unique(level='timestep'))  # number of timesteps
SAMPLES = len(shap_values.index.unique(level='sample'))  # number of samples

shap_min = data.shap_values.min()
shap_max = data.shap_values.max()

# %%
# Now, let's display the shap values for all features in each timestep.



# For each timestep (visualise all features)
steps = shap_values.index.get_level_values('timestep').unique()
for i, step in enumerate(steps):
    # Get interesting indexes
    indice = shap_values.index.get_level_values('timestep') == step

    # Create auxiliary matrices
    shap_aux = shap_values.iloc[indice]
    feat_aux = feature_values.iloc[indice]

    # Display
    plt.figure()
    plt.title("Timestep: %s" % step)
    shap.summary_plot(shap_aux.to_numpy(), feat_aux, show=False)
    plt.xlim(shap_min, shap_max)


# %%
# Now, let's display the shap values for all timesteps of each feature.

# For each feature (visualise all time-steps)
for i, f in enumerate(shap_values.columns[:N]):
    # Show
    # print('%2d. %s' % (i, f))

    # Create auxiliary matrices (select feature and reshape)
    shap_aux = shap_values.iloc[:, i] \
        .to_numpy().reshape(-1, TIMESTEPS)
    feat_aux = feature_values.iloc[:, i] \
        .to_numpy().reshape(-1, TIMESTEPS)
    feat_aux = pd.DataFrame(feat_aux,
        columns=['timestep %s' % j for j in range(-TIMESTEPS+1, 1)]
        )

    # Show
    plt.figure()
    plt.title("Feature: %s" % f)
    shap.summary_plot(shap_aux, feat_aux, sort=False, show=False)
    plt.xlim(shap_min, shap_max)
    plt.gca().invert_yaxis()

# Show
plt.show()
