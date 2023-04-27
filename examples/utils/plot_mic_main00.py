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

def component_information_v1(labels_true, labels_pred, *, contingency=None):
    """Computes the component information.

    The component information is calculated as below where X/Y
    denotes a state (e.g. RR).

        C(X/Y) = P(XY) * log(P(XY) / P(X)*P(Y))
               = P(XY) * [log(P(XY)) - log(P(X)*P(Y))]
               = P(XY) * [log(P(XY)) - (log(P(X)) + log(P(Y)))]
               = P(XY) * [log(P(XY)) - log(P(X)) - log(P(Y))]

    .. note: It is inspired by the code from sklearn.metrics.mutual_info_score.


    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    from scipy import sparse as sp
    from sklearn.metrics.cluster._supervised import check_clusterings
    from sklearn.metrics.cluster._supervised import check_array
    from sklearn.metrics.cluster._supervised import contingency_matrix

    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    else:
        contingency = check_array(
            contingency,
            accept_sparse=["csr", "csc", "coo"],
            dtype=[int, np.int32, np.int64],
        )

    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    elif sp.issparse(contingency):
        # For a sparse matrix
        nzx, nzy, nz_val = sp.find(contingency)
    else:
        raise ValueError("Unsupported type for 'contingency': %s" % type(contingency))

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    print("===> INSIDEEEE")
    print(nz_val)
    print(contingency_sum)
    print(pi)
    print(pj)
    # Since MI <= min(H(X), H(Y)), any labelling with zero entropy, i.e. containing a
    # single cluster, implies MI = 0
    if pi.size == 1 or pj.size == 1:
        return 0.0

    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * \
            pj.take(nzy).astype(np.int64, copy=False) # b*c
    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())


    print(log_contingency_nm)
    print(contingency_nm)
    print(outer)
    print(log_outer)
    mi = (
        contingency_nm * (log_contingency_nm - np.log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)

    return np.clip(mi.sum(), 0.0, None), mi



def component_info_score_v0(x=None, y=None, contingency=None):
    """

    The component information is calculated as:

        C(X/Y) = P(X/Y) * log(P(X/Y) / P(X)*P(Y))

    where X/Y is a state (e.g. RR).

    Thus

        e^(C(X/Y)) = e(

    Explanation
    -----------

          c11 c12
    C =   c21 c22

    pi = [pi1, pi2]
    pj = [pj1, pj2]

    ci11 = c11/n * np.log(c11/n / pi1*pj1)
    ci12 = c12/n * np.log(c12/n / pi1*pj2)
    ci21 = c21/n * np.log(c21/n / pi2*pj1)
    ci22 = c22/n * np.log(c22/n / pi2*pj2)

    """
    # Libraries
    import math

    if contingency is None:
        # Compute crosstab
        cts = pd.crosstab(x, y)
        cts['Total'] = cts.sum(axis=1)
        cts.loc['Total'] = cts.sum(numeric_only=True)
        cts = cts.copy() / cts.iloc[-1, -1]  # cross-tab with freqs

    # Continue
    if cts.shape[0] < 3:
        return None, None

    # Manuscript
    k = cts.to_numpy()
    s1 = k[0, 0] * np.log(k[0, 0] / k[0, 2] * k[2, 0])
    s2 = k[0, 1] * np.log(k[0, 1] / k[0, 2] * k[2, 1])
    s3 = k[1, 0] * np.log(k[1, 0] / k[1, 2] * k[2, 0])
    s4 = k[1, 1] * np.log(k[1, 1] / k[1, 2] * k[2, 1])

    if math.isnan(s1):
        s1 = 0
    if math.isnan(s2):
        s2 = 0
    if math.isnan(s3):
        s3 = 0
    if math.isnan(s4):
        s4 = 0

    cis = (s1 + s4) - (s2 + s3)

    # Return
    return cis, [s1, s2, s3, s4]

def component_info_score_v1(x=None, y=None, contingency=None):
    """Computes the component information.

    The component information is calculated as below where X/Y
    denotes a state (e.g. RR).

        C(X/Y) = P(X/Y) * log(P(X/Y) / P(X)*P(Y))

          c11 c12
    C =   c21 c22

    pi = [pi1, pi2]
    pj = [pj1, pj2]

    ci11 = c11/n * np.log(c11/n / pi1*pj1)
    ci12 = c12/n * np.log(c12/n / pi1*pj2)
    ci21 = c21/n * np.log(c21/n / pi2*pj1)
    ci22 = c22/n * np.log(c22/n / pi2*pj2)

    .. warning: It only works for square contingency matrices; that is, the
                number of different classes appearing in the vectors x and y
                must be the same.

                Enough for susceptibility test with only R and S. Will
                fail if only one is present or I is introduced.

    Example
    -------
    sex           Female  Male  Total
    lbl
    Not Survived      89   483    572
    Survived         230   112    342
    Total            319   595    914

    # Manual example.
    ci11 = (89  / 914) * np.log((89  / 914) / (572 / 914) * (319 / 914))
    ci12 = (483 / 914) * np.log((483 / 914) / (572 / 914) * (595 / 914))
    ci21 = (230 / 914) * np.log((230 / 914) / (342 / 914) * (319 / 914))
    ci22 = (112 / 914) * np.log((112 / 914) / (342 / 914) * (595 / 914))
    m = np.array([[ci11, ci12], [ci21, ci22]])

    # Compute
    m = component_information_v0(x=data.lbl, y=d.sex)

    Parameters
    ----------

    Returns
    -------

    """
    # Variables
    ct = crosstab(x, y)
    n = ct.count.sum()
    pi = np.ravel(ct.count.sum(axis=1))
    pj = np.ravel(ct.count.sum(axis=0))

    # Compute
    b = np.repeat(pi.reshape(-1,1), len(pi), axis=1)
    c = np.repeat(pj.reshape(1,-1), len(pj), axis=0)
    m = (ct.count/n) * np.log((ct.count/n) / (b/n) * (c/n))

    # Return
    return None, m

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
#                       Manual
# ----------------------------------------------------------
# Definition of MI from <sklearn>:
# """The Mutual Information (mutual_information_score) measures the similarity
# between two labels of the same data. This metric is independent of the absolute
# values of the labels: a permutation of the class or cluster label values won’t
# change the score value in any way."""
#
# Definition of MI from <manuscript>
# """Mutual information quantifies the degree of dependence between two antibiotic
# susceptibility test results (X and Y) by measuring the amount of information
# gained about one test result (X) by knowing that of the other (Y). Because mutual
# information captures the deviation from independence between the individual
# antibiotics of a pair, it is well suited for identifying disjoint resistance"""
#
# MIS = (CI_SS + CI_RR) - (CI_SR + CI_RS)
#
# .. note: For the vectors [R, R, R, R] and [S, S, S, S] the mutual information
#          score is 0.0. Since the labels are completely opposite this makes sense
#          for the sklearn definition. But since knowing one test result completely
#          explains the other, it does not agree with the manuscript definition.
#          Is this correct?
#
# https://rdrr.io/bioc/maigesPack/man/MI.html
# Computing mis5 online gives 1.506118

mis0 = mutual_info_score(
    ['R', 'R', 'R', 'R'],
    ['S', 'S', 'S', 'S']
)
mis1 = mutual_info_score(
    ['S', 'S', 'S', 'R'],
    ['S', 'S', 'R', 'S']
)
mis2 = mutual_info_score(
    ['R', 'S', 'R', 'R'],
    ['S', 'S', 'S', 'R']
)
mis3 = mutual_info_score(
    ['R', 'R', 'R', 'R'],
    ['R', 'S', 'R', 'R']
)
mis4 = mutual_info_score(
    ['R', 'R', 'R', 'R'],
    ['R', 'R', 'R', 'R']
)
mis5 = mutual_info_score(
    [1, 5, 4, 9, 0],
    [1, 5, 4, 9, 0]
)
mis6 = mutual_info_score(
    [0, 0, 0, 1],
    [0, 0, 1, 0]
)

show = [mis0, mis1, mis2, mis3, mis4, mis5, mis6]

print("\nManual examples:")
for i, e in enumerate(show):
    print("%2s. MIS: %.5f" % (i, e))
print("\n"+"="*80)





# ----------------------------------------------------------
#                       Manual
# ----------------------------------------------------------
#
def create_data_product(query=None,
                        choices=['R', 'S'],
                        size=4):
    """"""
    # Library
    import itertools
    # Create random query
    if query is None:
        query = np.random.choice(choices, size=size)
    # Create tuples and repeat query.
    t = [p for p in itertools.product(choices, repeat=len(query))]
    q = [query for i in range(len(t))]
    # Return
    return list(zip(q, t))

data1 = [
    (['R', 'R', 'R', 'R'], ['S', 'S', 'S', 'S']),
    (['S', 'S', 'S', 'R'], ['S', 'S', 'R', 'S']),
    (['R', 'S', 'R', 'R'], ['S', 'S', 'S', 'R']),
    (['R', 'R', 'R', 'R'], ['R', 'S', 'R', 'R']),
    (['R', 'R', 'R', 'R'], ['R', 'R', 'R', 'R']),
    ([1, 5, 4], [1, 5, 4]),
    ([0, 0, 0, 1], [0, 0, 1, 0])
]

data2 = [
    (['R', 'R', 'R', 'R'], ['R', 'R', 'R', 'R']),
    (['R', 'R', 'R', 'R'], ['S', 'S', 'S', 'S']),
    (['R', 'R', 'R', 'R'], ['R', 'R', 'S', 'S'])
]

# Create product data.
data3 = create_data_product(query=['R', 'R', 'R', 'R'])

#
data = data1

# Compute the mutual information score.
print("\nMutual Information Score:")
for x, y in data:
    s = mutual_info_score(x, y)
    print("%20s | %20s | %.5f" % (x, y, s))

# Compute the component information score.
#print("\nComponent Information Score:")
#for x, y in data:
#    _, m = component_info_score_v0(x, y)
#    print("%20s | %20s | %s" % (x, y, m))

# Compute the component information score.
#print("\nComponent Information Score:")
#for x, y in data:
#    _, m = component_info_score_v1(x, y)
#    print("%20s | %20s | %s" % (x, y, m.round(4).tolist()))


print("\nComponent Information Score:")
for x, y in data:
    m = component_info_score_v3(x, y)
    print("%20s | %20s | %s" % (x, y, m.round(4).tolist()))


import sys
sys.exit()

"""
# ----------------------------------------------------------
#                       Example 0
# ----------------------------------------------------------
#
# .. note: why the mutual information score is always 0 or 1?
#
# Libraries
import itertools

# Create query and possible combinations
choices = ['R', 'S']
q = ['R', 'R', 'R']
q = np.random.choice(choices, size=4)
t = [p for p in itertools.product(choices, repeat=len(q))]

# Loop computing mutual information score
for i,x in enumerate(t):
    print("\n%2s/%2s. Loop" % (i, len(t)))
    ctm = crosstab(q, x)
    ctm = pd.crosstab(q, x)

    # Compute mutual information score
    mis = mutual_info_score(labels_true=q,
                            labels_pred=x)

    # Compute component information score
    cis, cim = component_info_score_v1(q, x)

    # Show
    #print(ctm)
    print("Q: %s" % str(q))
    print("X: %s" % str(list(x)))
    print("MIS: %s" % (mis))
    #print("CIM: %s" % cim.tolist())
"""


# ----------------------------------------------------------
#                       Example 1
# ----------------------------------------------------------
# Generate the dataset
x = np.array([
    ['Female', 'Not Survived'],
    ['Female', 'Survived'],
    ['Male', 'Not Survived'],
    ['Male', 'Survived']])
d = np.repeat(x, [89, 230, 483, 112], axis=0)
d = pd.DataFrame(data=d, columns=['sex', 'lbl'])

# Show
print("\nData:")
print(d)

# Compute distributions
ct = pd.crosstab(d.lbl, d.sex)       # cross-tab
cts = ct.copy()                      # cross-tab with sums
cts['Total'] = cts.sum(axis=1)
cts.loc['Total'] = cts.sum(numeric_only=True)
ctf = cts.copy() / cts.iloc[-1, -1]  # cross-tab with freqs

# Show
print("\nContingency matrix (freqs):")
print(cts)
print("\nContingency matrix (probs):")
print(ctf)

#  I(X,Y) = sum_x( sum_y( p(x,y) * log( p(x,y) / p(x)*p(y) )))

# Compute mutual information score
mis1 = mutual_info_score(labels_true=None,
                         labels_pred=None,
                         contingency=ct)
mis2 = mutual_info_score(
    labels_true=d.sex, labels_pred=d.lbl)
mis3 = adjusted_mutual_info_score(
    labels_true=d.sex, labels_pred=d.lbl)
mis4 = normalized_mutual_info_score(
    labels_true=d.sex, labels_pred=d.lbl)

# Show
print("\nMutual Information (MI) Scores:")
print("MIS (vectors):     %.3f" % mis1)
print("MIS (contingency): %.3f" % mis2)
print("MIS (adjusted):    %.3f" % mis3)
print("MIS (normalized):  %.3f" % mis4)

# Manual
ci11 = (89 / 914) * np.log((89 / 914) / (572 / 914) * (319 / 914))
ci12 = (483 / 914) * np.log((483 / 914) / (572 / 914) * (595 / 914))
ci21 = (230 / 914) * np.log((230 / 914) / (342 / 914) * (319 / 914))
ci22 = (112 / 914) * np.log((112 / 914) / (342 / 914) * (595 / 914))
m = np.array([[ci11, ci12], [ci21, ci22]])
print("\nCI scores:")
print(None)
print(m)

cis, cic = component_info_score_v3(x=d.lbl, y=d.sex)
print(cic)


cis, cic = component_information_v3(x=d.lbl, y=d.sex)
print("\nCI scores:")
print(cis)
print(cic)

cis, cic = component_information_v0(x=d.lbl, y=d.sex)
print("\nCI scores:")
print(cis)
print(cic)

cis, cic = component_information_v1(
        labels_true=d.lbl,
        labels_pred=d.sex)

print("\nCI scores:")
print(cis)
print(cic)

import sys
sys.exit()

# ----------------------------------------------------------
#                       Example 2
# ----------------------------------------------------------
# Data
a = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
x = ['X', 'X', 'X', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z']

cis, cic = component_information_v0(x=a, y=x)
print("\nCI scores:")
print(cis)
print(cic)

import sys
sys.exit()

# ----------------------------------------------------------
#                       Example 2
# ----------------------------------------------------------
# Data
a = ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
x = ['X', 'X', 'X', 'Y', 'Z', 'Z', 'Y', 'Y', 'Z', 'Z']

# Compute score
mis1 = mutual_info_score(a,x)

c = crosstab(a, x)
mis2 = mutual_info_score(labels_true=None,
                         labels_pred=None,
                         contingency=c[1])

# Show
print("\nMutual Information (MI) Scores:")
print("MIS (vectors):     %.3f" % mis1)
print("MIS (contingency): %.3f" % mis2)


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
    columns=['sample'] + ['abx_%s'%i for i in range(8)]
)

"""
for c in list(combinations(range(6), 2)):
    x = data['abx_%s'%c[0]]
    y = data['abx_%s'%c[1]]
    data['class_%s_%s'% c ] = x + y
"""
print("\nOther:")
print(data)




combos = list(combinations(range(8), 2))
for c in combos:
    x = data['abx_%s'%c[0]]
    y = data['abx_%s'%c[1]]

    # Compute mutual information score
    m = mutual_info_score(x, y)
    m1 = adjusted_mutual_info_score(x, y)
    m2 = normalized_mutual_info_score(x, y)

    # Compute crosstab
    #cts = pd.crosstab(x, y)
    #cts['Total'] = cts.sum(axis=1)
    #cts.loc['Total'] = cts.sum(numeric_only=True)

    #if cts.shape[0] < 3:
    #    continue

    cis, cic = component_information(x=x, y=y)

    if cis is None:
        continue


    # Show
    print(c)
    #print(cts)
    print('MIS=%.2f | MIS_A=%.2f | MIS_N=%.2f' % (m, m1, m2))
    print('RR=%.2f | RS=%.2f | SR=%.2f | SS=%.2f' % (cic[0], cic[1], cic[2], cic[3]))
    print('CIS=%.2f' % cis)
    print("\n\n")

