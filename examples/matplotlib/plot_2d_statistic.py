import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def data_random(N=100):
    """"""
    # Create random values
    x = np.random.randint(-10, 0, size=N)
    y = x + np.random.normal(size=N) * (x / 10)
    z = np.random.normal(size=N)**x
    return x, y, z

def data_shap():
    """"""
    data = pd.read_csv('./data/shap.csv')
    print(data)
    return data.timestep, data.feature_values, None


# Load data
x, y, z = data_random()
x, y, z = data_shap()

# Compute binned statistic
statistic, x_edge, y_edge, binnumber = \
    stats.binned_statistic_2d(x=x, y=y, values=None,
        statistic='count', bins=[11, 10],
        expand_binnumbers=True)

statistic, x_edge, y_edge, binnumber = \
    stats.binned_statistic_2d(x=x, y=y, values=None,
        statistic='count', bins=[5, 5],
        expand_binnumbers=False)

# Compute bin centers
x_center = (x_edge[:-1] + x_edge[1:]) / 2
y_center = (y_edge[:-1] + y_edge[1:]) / 2

# Show
print(statistic)
print(x_edge)
print(y_edge)
print(binnumber)

# Display
plt.imshow(statistic)
#plt.xticks(x_center)
#plt.yticks(y_center)
plt.show()


"""
x = [0.1, 0.1, 0.1, 0.6]
y = [2.1, 2.6, 2.1, 2.1]
binx = [0.0, 0.5, 1.0]
biny = [2.0, 2.5, 3.0]
ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny])


ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny],
                                expand_binnumbers=True)
ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny],
                                expand_binnumbers=False)
print(ret)
"""