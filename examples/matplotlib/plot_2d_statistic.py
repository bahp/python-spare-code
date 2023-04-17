from scipy import stats

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