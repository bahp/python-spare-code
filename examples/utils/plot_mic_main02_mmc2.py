# Libraries
import numpy as np
import pandas as pd

from itertools import combinations
from timeit import default_timer as timer
from scipy.stats.contingency import crosstab
from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

from mic import component_info_score_v3
from mic import component_info_score_2d



# ------------------------------------------------------------
#                   Example 1: Hardcoded
# ------------------------------------------------------------
# See: https://www.youtube.com/watch?v=eJIp_mgVLwE
# This comes from a youtube example. The most important things
# to highlight from this code are as follows:
#   (i) log(0) raises a zero division.
#  (ii) lim x->0 log(x) = 0
# (iii) this np.nan can be filled with 0.

# Contingency
ct = np.array([[3/5, 1/5], [0/5, 1/5]])

# Compute MIS manually
mi1 = (3/5)*np.log((3/5) / ((3/5)*(4/5)))
#mi2 = (0/5)*np.log((0/5) / ((3/5)*(1/5))) # zero division
mi3 = (1/5)*np.log((1/5) / ((2/5)*(4/5)))
mi4 = (1/5)*np.log((1/5) / ((2/5)*(1/5)))
m1 = np.array([[mi1, mi3], [0, mi4]])
score1 = mi1 + mi3 + mi4 # 0.22

# Compute component information matrix
score2, m2 = component_info_score_v3(ct=ct)

# Compute component information matrix
score3, m3 = component_info_score_2d(ct=ct)

# .. note: Raises a math domain error.
# Compute MIS scikits
#score4 = mutual_info_score(labels_true=None,
#                           labels_pred=None,
#                           contingency=ct)

# Show
print("\n" + "="*80)
print("Example 1")
print("="*80)
print("\nContigency:")
print(ct)
print("\nManual:")
print(m1)
print(score1)
print("\nMethod v3:")
print(m2)
print(m2.sum())
print("\nMethod 2d:")
print(m3)
print(m3.sum())
#print("\nScikits:")
#print(score4)



# ------------------------------------------------------------
#                     Example 2: Harcoded
# ------------------------------------------------------------
# This is same as before but now we create the whole vectors.

# Generate the dataset
x = np.array([
    ['S1', 'S2'],
    ['S1', 'R2'],
    ['R1', 'S2'],
    ['R1', 'R2']])
d = np.repeat(x, [63, 22, 15, 25], axis=0)
d = pd.DataFrame(data=d)

# Compute contingency
ct = crosstab(d[0], d[1]).count
ct_pandas = pd.crosstab(d[0], d[1])

# Compute MIS
score0 = mutual_info_score(labels_true=d[0],
                           labels_pred=d[1])

# Compute MIS
score1, m1 = component_info_score_v3(d[0], d[1])

# Compute MIS
score2, m2 = component_info_score_v3(ct=ct)

# Compute MIS
score3, m3 = component_info_score_2d(d[0], d[1])

# Display
print("\n" + "="*80)
print("Example 2")
print("="*80)
print('\nData:')
print(d)
print('\nContingency:')
print(ct_pandas)
print('\nMIS (Scikits):')
print(score0)
print('\nMethod v3:')
print(m1)
print(score1)
print(m1.sum())
print('\nMethod v3:')
print(m2)
print(score2)
print(m2.sum())
print('\nMethod 2d:')
print(m3)
print(score3)
print(m3.sum())


# ------------------------------------------------------------
#                     Example 3: File
# ------------------------------------------------------------
# Compute the MIS score as defined in the manuscript using
# the manuscript provided cumulative data to see whether
# the results match.

def MIS_v3(x):
    ct = np.array([[x.S1S2, x.S1R2], [x.R1S2, x.R1R2]])
    score, m = component_info_score_v3(ct=ct)
    return score

def MIS_2d(x):
    ct = np.array([[x.S1S2, x.S1R2], [x.R1S2, x.R1R2]])
    score, m = component_info_score_2d(ct=ct)
    return score

# Load data
data = pd.read_excel('./data/mmc2.xlsx')

# Compute MIC score ourselves
data['MIS_v3'] = data.apply(MIS_v3, axis=1)

# Compute MIC score ourselves
data['MIS_2d'] = data.apply(MIS_2d, axis=1)

print("\n" + "="*80)
print("Example 3")
print("="*80)
print(data)


# ------------------------------------------------------------
#                     Example 3: timing
# ------------------------------------------------------------
# This code is used to compare whether the implementations are
# more or less efficient between them. Note that the methods
# have itself some limitations.

# Generate data
N = 1000
choices = np.arange(100)
vector1 = np.random.choice(choices, size=N)
vector2 = np.random.choice(choices, size=N)

t1 = timer()
score1, m1 = component_info_score_v3(
    x=vector1, y=vector2
)
t2 = timer()
score2, m2 = component_info_score_2d(
    x=vector1, y=vector2
)
t3 = timer()

# Display
print("\n" + "="*80)
print("Example 4")
print("="*80)
print("Are the results equal? %s" % np.array_equal(m1, m2))
print("time v3: %.5f" % (t2-t1))
print("time 2d: %.5f" % (t3-t2))





"""
# ------------------------------------------------------------
#                   Example 2: Hardcoded
# ------------------------------------------------------------
# How to manually compute the mutual information score when
# there are only two classes (e.g. R, S). In particular the
# values included correspond to the row in mmc2.xls for the
# following drugs:
#
#   AMOXICILLIN_CLAVULANATE
#   AZTREONAM

# Variables
c = np.array([[25, 15], [22, 63]])
c = np.array([[89, 483], [230, 112]])
c = np.array([[3/5, 1/5], [0/5, 1/5]])
n = c.sum()
pi = np.ravel(c.sum(axis=1))
pj = np.ravel(c.sum(axis=0))

# Manual example of MIS.
ci11 = (c[0,0] / n) * np.log((c[0,0] / n) / (pi[0] / n) * (pj[0] / n))
ci12 = (c[0,1] / n) * np.log((c[0,1] / n) / (pi[0] / n) * (pj[1] / n))
ci21 = (c[1,0] / n) * np.log((c[1,0] / n) / (pi[1] / n) * (pj[0] / n))
ci22 = (c[1,1] / n) * np.log((c[1,1] / n) / (pi[1] / n) * (pj[1] / n))
m = np.array([[ci11, ci12], [ci21, ci22]])
score1 = (m[0,0] + m[1,1]) - (m[0,1] + m[1,0])

aux = np.copy(m)
aux[np.isnan(aux)] = 0
score2 = aux.sum()
score3 = (aux[0,0] + aux[1,1]) - (aux[0,1] + aux[1,0])

# compute
#mis1 = mutual_info_score(labels_true=False,
#                         labels_pred=False,
#                         contingency=c)

print("\n" + "="*80)
print("Example 1")
print("="*80)
print('\nContingency:')
print(c)
print('\nMutual Information Score:')
#print(mis1)
print('\nMutual Information Score:')
print(m)
print(score1)
print(score2)
print(score3)
"""