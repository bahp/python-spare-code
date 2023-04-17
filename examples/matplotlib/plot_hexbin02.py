import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

# Load data
data = pd.read_csv('./data/shap.csv')
x = data.timestep
y = data.shap_values
z = data.feature_values

# Compute limits
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

# Display hexagon binning (linear)
fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(7, 4))
fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
hb = ax.hexbin(x, y, C=z, gridsize=[7, 30], cmap='coolwarm', reduce_C_function=np.median)
ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("Hexagon binning")
#ax.invert_xaxis()
cb = fig.colorbar(hb, ax=ax)
cb.set_label('N=median')

# Show
plt.tight_layout()
plt.show()

"""
.. note: Needs revision

H, yedges, xedges = np.histogram2d(y, x, bins=20)

# Plot histogram using pcolormesh
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
ax1.pcolormesh(xedges, yedges, H, cmap='coolwarm')
#ax1.plot(x, y, 'k-')
ax1.set_xlim(x.min(), x.max())
ax1.set_ylim(y.min(), y.max())
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('histogram2d')
ax1.grid()
ax1.invert_xaxis()

# Create hexbin plot for comparison
ax2.hexbin(x, y, gridsize=20, cmap='coolwarm')
#ax2.plot(x, 2*np.log(x), 'k-')
ax2.set_title('hexbin')
ax2.set_xlim(x.min(), x.max())
ax2.set_ylim(y.min(), y.max())
ax2.set_xlabel('x')
ax2.grid()
ax2.invert_xaxis()
"""
