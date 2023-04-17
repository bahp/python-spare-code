import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def data_random(N=100):
    """"""
    # Create random values
    x = np.random.randint(-10, 0, size=N)
    y = x + np.random.normal(size=N) * (x / 10)
    return x, y

def data_shap():
    """"""
    data = pd.read_csv('./data/shap.csv')
    print(data)
    return data.timestep, data.feature_values



# ----------------------------------------------
# Main
# ----------------------------------------------
#binx = np.linspace(-10, 0, 11)
#biny = np.linspace(-1, 1, 100)

# Load data
#x, y = data_shap()
x, y = data_random()

# Compute bins
bin_means, bin_edges, binnumber = \
    stats.binned_statistic(x=x, values=y,
        statistic='median', bins=10)

# Show
print(bin_means)
print(bin_edges)
print(binnumber)

# Display
plt.figure()
plt.plot(x, y, 'b.', label='raw data')
plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:],
    colors='g', lw=5, label='binned statistic of data')
plt.legend()



# Compute bins
bin_means, bin_edges, binnumber = \
    stats.binned_statistic_2d(x=x, values=y,
        statistic='median', bins=10,
        expand_binnumbers=False)

print(binnumber)

plt.show()



"""
# Example Boat
# ------------
rng = np.random.default_rng()
windspeed = 8 * rng.random(500)
boatspeed = .3 * windspeed**.5 + .2 * rng.random(500)
bin_means, bin_edges, binnumber = stats.binned_statistic(windspeed,
                boatspeed, statistic='median', bins=[1,2,3,4,5,6,7])
plt.figure()
plt.plot(windspeed, boatspeed, 'b.', label='raw data')
plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
           label='binned statistic of data')
plt.legend()
plt.show()
import sys
sys.exit()



x = [0.1, 0.1, 0.1, 0.6]
y = [2.1, 2.6, 2.1, 2.1]
binx = [0.0, 0.5, 1.0]
biny = [2.0, 2.5, 3.0]
#ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny])


#ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny],
#                                expand_binnumbers=True)
ret = stats.binned_statistic_2d(x, y, None,
    statistic='mean', bins=[binx, biny],
    expand_binnumbers=False)
print(ret)
print(ret.binnumber)

# Show
#plt.imshow(ret.binnumber)
"""