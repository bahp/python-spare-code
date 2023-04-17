import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_random(n=100000):
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    # Generate data
    x = np.random.standard_normal(n)
    y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
    return x, y, None

def data_shap():
    data = pd.read_csv('./data/shap.csv')
    return data.timestep, data.shap_values, data.feature_values


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
# Load data
#x, y, c = data_random()
x, y, c = data_shap()

# Compute limits
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

# Display hexagon binning (linear)
fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
ax = axs[0]
hb = ax.hexbin(x, y, C=c, cmap='coolwarm', reduce_C_function=np.median)
ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("Hexagon binning")
ax.invert_xaxis()
cb = fig.colorbar(hb, ax=ax)
cb.set_label('N=median')

# Display hexagon binning (log)
ax = axs[1]
hb = ax.hexbin(x, y, C=c, gridsize=50, bins='log', cmap='coolwarm', reduce_C_function=np.median)
ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("With a log color scale")
ax.invert_xaxis()
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(N)')

# Show
plt.tight_layout()
plt.show()