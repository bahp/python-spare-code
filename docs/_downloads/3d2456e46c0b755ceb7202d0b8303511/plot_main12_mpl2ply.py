"""
12a. Matplotlib to Plotly (errorbar)
=====================================

This example converts a matplotlib figure to Plotly.

.. warning:: A known bug in Plotly (see GitHub issue) causes the mpl_to_plotly() function
             to fail because it references an outdated function from Matplotlib.
             The current fix is to downgrade to an older version of Matplotlib or to
             recreate your figure in Plotly manually.

             [ISSUE]: https://github.com/plotly/plotly.py/issues/3624#issuecomment-1161805210
             In the latest commit of plotly packages/python/plotly/plotly/matplotlylib/mpltools.py
             line 368, it still calls is_frame_like() function.

"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False


# Create data
x = np.arange(10)
y = 2.5 * np.sin(x / 20 * np.pi)
yerr = np.linspace(0.05, 0.2, 10)

# Figure
fig = plt.figure()
plt.errorbar(x, y + 3, yerr=yerr, label='both limits (default)')
plt.errorbar(x, y + 2, yerr=yerr, uplims=True, label='uplims=True')
plt.errorbar(x, y + 1, yerr=yerr, uplims=True, lolims=True,
             label='uplims=True, lolims=True')

# Add error
upperlimits = [True, False] * 5
lowerlimits = [False, True] * 5
plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits,
             label='subsets of uplims and lolims')

# add legent
plt.legend(loc='lower right')


"""
.. note:: This section has been commented as it does not work
          with recent versions of matplotlib. Downgrade an
          remove quotes to run it.
          
# Convert to plotly
import plotly.tools as tls

# Convert to plotly
fig = tls.mpl_to_plotly(fig)

print(fig.data)

# Show
#fig.show()
fig
"""
