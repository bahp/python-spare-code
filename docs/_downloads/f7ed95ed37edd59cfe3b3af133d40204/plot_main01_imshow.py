"""
01. 2D array with ``mpl.imshow``
================================

This script provides a basic example of using ``matplotlib.imshow``
to visualize a 2D NumPy array as a raster image. 🖼️ It generates
random data and displays it as a plot, then adds a shared
colorbar to show the value-to-color mapping.
"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# Display
plt.subplot(211)
plt.imshow(np.random.random((100, 100)))
plt.subplot(212)
plt.imshow(np.random.random((100, 100)))

# Adjust
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.tight_layout()

# Show
plt.show()