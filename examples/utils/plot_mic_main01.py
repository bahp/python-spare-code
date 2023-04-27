"""
Mutual Information Criteria
---------------------------





v1 = [R, R, R, R]
v2 = [R, R, R, R]


"""
# 1) https://www.blog.trainindata.com/mutual-information-with-python/
# 2) https://stats.stackexchange.com/questions/531527/mutual-information-and-maximual-infomation-coffecient
# 3) https://stats.stackexchange.com/questions/306131/how-to-correctly-compute-mutual-information-python-example
# 4) https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy

# Libraries
import pandas as pd
import numpy as np

from itertools import combinations
from scipy.stats.contingency import crosstab
from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

def component_info_score_v3(x, y):
    """Compute the component information score.

    .. note: Very inefficient but good for testing.

    Parameters
    ----------
    x: list
        List with the classes
    y: list
        List with the classes

    Returns
    -------
    """
    # Variables
    ct = crosstab(x, y)
    n = ct.count.sum()
    pi = np.ravel(ct.count.sum(axis=1))
    pj = np.ravel(ct.count.sum(axis=0))

    # Create empty matrix
    m = np.empty(ct.count.shape)
    m[:] = np.nan

    # Fill with component information score
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if ct.count[i,j] == 0:
                continue
            pxy = ct.count[i,j] / n
            m[i,j] = pxy * np.log(pxy / pi[i] * pj[j])

    # Return
    return m



# ----------------------------------------------------------
#                       Example 3
# ----------------------------------------------------------
# Data
data = [
    ['S1', 'R', 'R', 'S', 'S', 'R', 'S', 'S', 'R'],
    ['S2', 'R', 'R', 'S', 'S', 'S', 'R', 'S', 'R'],
    ['S3', 'R', 'R', 'S', 'R', 'R', 'S', 'S', 'R'],
    ['S4', 'R', 'R', 'S', 'R', 'S', 'R', 'R', 'S'],
    ['S5', 'R', 'R', 'S', 'S', 'R', 'S', 'R', 'R'],
    ['S6', 'R', 'R', 'S', 'S', 'S', 'R', 'R', 'S'],
    ['S7', 'R', 'R', 'S', 'R', 'R', 'S', 'R', 'S'],
    ['S8', 'R', 'R', 'S', 'R', 'S', 'R', 'R', 'R'],
]
data = pd.DataFrame(
    data=data,
    columns=['sample'] + ['abx%s'%i for i in range(8)]
)

"""
for c in list(combinations(range(6), 2)):
    x = data['abx_%s'%c[0]]
    y = data['abx_%s'%c[1]]
    data['class_%s_%s'% c ] = x + y
"""
print("\nOther:")
print(data)


mim = component_info_score_v3(data.abx1, data.abx2)
print(mim)
mim = component_info_score_v3(data.abx1, data.abx3)
print(mim)
mim = component_info_score_v3(data.abx1, data.abx4)
print(mim)
mim = component_info_score_v3(data.abx1, data.abx6)
print(mim)



combos = list(combinations(range(8), 2))
for c in combos:
    x = data['abx%s'%c[0]]
    y = data['abx%s'%c[1]]

    # Compute mutual information score
    m0 = mutual_info_score(x, y)
    m1 = adjusted_mutual_info_score(x, y)
    m2 = normalized_mutual_info_score(x, y)

    # Compute component information score
    mim = component_info_score_v3(x=x, y=y)

    # Compute CIS
    cis = -100
    if mim.shape[0] > 1:
        cis = (mim[0,0] + mim[1,1]) - (mim[0,1] + mim[1,0])

    # Show
    print("\n")
    print(x.tolist())
    print(y.tolist())
    print('MIS=%.2f | MIS_A=%.2f | MIS_N=%.2f' % (m0, m1, m2))
    print('CIS=%.2f' % cis)


    #if mim.shape[0]>1:
    #print(mim)
    #step = len(mim) - 1;
    #d = np.take(mim, np.arange(step, mim.size-1, step))

    #print(d)


    # np.arange(step, array.size-1, step)

    #if cis is None:
    #    continue

    print(mim)
    # Show
    #print(c)
    #print(cts)
    #print('MIS=%.2f | MIS_A=%.2f | MIS_N=%.2f' % (m, m1, m2))
    #print('RR=%.2f | RS=%.2f | SR=%.2f | SS=%.2f' % (cic[0], cic[1], cic[2], cic[3]))
    #print('CIS=%.2f' % cis)
    #print("\n\n")
