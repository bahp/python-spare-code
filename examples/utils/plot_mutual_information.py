"""
Mutual Information Criteria
---------------------------

The ``Mutual Information Score``, often denoted as ``MIS``, expresses the extent
to which observed frequency of co-occurrence differs from what we would expect
(statistically speaking). In statistically pure terms this is a measure of the
strength of association between words x and y.

See below for a few resources.

  * `R1`_: Detailed video tutorial step by step.
  * `R2`_: Detailed tutorial with python code.
  * `R3`_: Possible libraries in python/R and other tools.
  * `R5`_: Efficient pairwise MIS implementation...

.. _R1: https://www.youtube.com/watch?v=eJIp_mgVLwE
.. _R2: https://www.blog.trainindata.com/mutual-information-with-python/
.. _R3: https://stats.stackexchange.com/questions/531527/mutual-information-and-maximual-infomation-coffecient
.. _R4: https://stats.stackexchange.com/questions/306131/how-to-correctly-compute-mutual-information-python-example
.. _R5: https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy

.. _M1: https://www.thelancet.com/journals/lanmic/article/PIIS2666-5247(21)00118-X/fulltext

"""

#######################################################################
# Lets import the main libraries

# Generic
import warnings
import numpy as np
import pandas as pd

# Specific
from itertools import combinations
from timeit import default_timer as timer
from scipy.stats.contingency import crosstab
from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

# Own
from mic import mutual_info_matrix_v3
from mic import mutual_info_matrix_v2
from mic import mutual_info_matrix_v1

warnings.filterwarnings("ignore")

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False

######################################################################
# a) Manual example (youtube)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lets start with a hard coded example extracted from a very
# detailed youtube tutorial (`R1`_). This video tutorial shows step by
# step how to compute the mutual information score using the contingency
# matrix defined below. Pay special attention to the following consideration
# when implementing the ``MIS``:
#
#   * only possible to compute where more than one class present
#   * log(0) raises a zero division.
#   * lim x->0 log(x) = 0
#   * this np.nan can be filled with 0.
#

# See: https://www.youtube.com/watch?v=eJIp_mgVLwE

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
m2 = mutual_info_matrix_v1(ct=ct)
m3 = mutual_info_matrix_v2(ct=ct)
m4 = mutual_info_matrix_v3(ct=ct)

# .. note: Raises a math domain error.
# Compute MIS scikits
#score4 = mutual_info_score(labels_true=None,
#                           labels_pred=None,
#                           contingency=ct)

# Cumu
cumu = pd.DataFrame([
    ['manual'] + m1.flatten().tolist(),
    ['mutual_info_matrix_v1'] + m2.flatten().tolist(),
    ['mutual_info_matrix_v2'] + m3.flatten().tolist(),
    ['mutual_info_matrix_v3'] + m4.flatten().tolist()
], columns=['method', 'c11', 'c12', 'c21', 'c22'])

# Compute MIS score
cumu['mis'] = cumu.sum(axis=1)

#%%
# Lets see the contingency matrix
if TERMINAL:
    print("\n" + "="*80 + "\nExample 1\n" + "="*80)
    print('\nContingency:')
    print(ct)
pd.DataFrame(ct)

#%%
# Lets see the results
if TERMINAL:
    print("\nResults:")
    print(cumu)
cumu


#%%
# .. note:: The method ``mutual_info_matrix_v1`` does not work in this example!


######################################################################
# b) Another two class example
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In the previous example we started with the definition of the contingency
# matrix. However, that is not often the case. In this example we will go
# one step back and show how to compute the contingency matrix from the raw
# vectors using either scipy o pandas. Note that the contingency matrix is
# just a way to display the frequency distribution of the variables.

# Generate the dataset
x = np.array([
    ['S1', 'S2'],
    ['S1', 'R2'],
    ['R1', 'S2'],
    ['R1', 'R2']])
d = np.repeat(x, [63, 22, 15, 25], axis=0)
d = pd.DataFrame(data=d)

# Create variables
x = d[0]
y = d[1]

# Compute contingency
#ct = crosstab(d[0], d[1]).count
ct = pd.crosstab(x, y)

# Compute MIS
score0 = mutual_info_score(labels_true=x, labels_pred=y)

# Compute MIS
m1 = mutual_info_matrix_v1(x=x, y=y)
m2 = mutual_info_matrix_v2(x=x, y=y)
m3 = mutual_info_matrix_v3(x=x, y=y)

# Compute MIS
m4 = mutual_info_matrix_v1(ct=ct)
m5 = mutual_info_matrix_v2(ct=ct)
m6 = mutual_info_matrix_v3(ct=ct)

# Cumu
cumu = pd.DataFrame([
    #['mutual_info_score'] + m1.flatten().tolist(),
    ['mutual_info_matrix_v1 (x,y)'] + m1.flatten().tolist(),
    ['mutual_info_matrix_v2 (x,y)'] + m2.flatten().tolist(),
    ['mutual_info_matrix_v3 (x,y)'] + m3.flatten().tolist(),
    ['mutual_info_matrix_v1 (ct)'] + m4.flatten().tolist(),
    ['mutual_info_matrix_v2 (ct)'] + m5.flatten().tolist(),
    ['mutual_info_matrix_v3 (ct)'] + m6.flatten().tolist(),
], columns=['method', 'c11', 'c12', 'c21', 'c22'])

# Compute MIS score
cumu['mis'] = cumu.sum(axis=1)

#%%
# Lets see the contingency matrix
if TERMINAL:
    print("\n" + "="*80 + "\nExample 2\n" + "="*80)
    print('\nContingency:')
    print(ct)
ct

#%%
# Lets see the results
if TERMINAL:
    print("\nResults:")
    print(cumu)
cumu


########################################################################
# c) Collateral Resistance Index
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now, lets compute the ``MIS`` score as defined in the manuscript (`R5`_).
# Note that the manuscript provided the cumulative data as appendix
# material and therefore we can use it to compare that our implementation
# produces the same result.
#
# .. note:: The results provided by our own ``MIS`` implementation differs
#           from the results provided in the manuscript. This discrepancy
#           occurs for those rows in which the contingency matrix contains
#           one of more zeros.

def collateral_resistance_index(m):
    """Collateral Resistance Index

    The collateral resistance index is based on the mutual
    information matrix. This implementation assumes there
    are two classes resistant (R) and sensitive (S).

    Parameters
    ----------
    m: np.array
        A numpy array with the mutual information matrix.

    Returns
    -------
    """
    return (m[0, 0] + m[1, 1]) - (m[0, 1] + m[1, 0])

def CRI(x, func):
    ct = np.array([[x.S1S2, x.S1R2], [x.R1S2, x.R1R2]])
    m = func(ct=ct)
    return collateral_resistance_index(m)

def compare(data, x, y):
    return data[x].round(5).compare(data[y].round(5)).index.values

# Load data
data = pd.read_excel('./data/mmc2.xlsx')

# Compute MIC score ourselves
#data['MIS_v1'] = data.apply(CRI, args=(mutual_info_matrix_v1,), axis=1)
data['MIS_v2'] = data.apply(CRI, args=(mutual_info_matrix_v2,), axis=1)
data['MIS_v3'] = data.apply(CRI, args=(mutual_info_matrix_v3,), axis=1)

# Compute indexes of those that do not give same result.
idxs1 = compare(data, 'MIS', 'MIS_v3')

#%%
# Lets see the data
if TERMINAL:
    print("\n" + "=" * 80 + "\nExample 3\n" + "=" * 80)
    print("\nData:")
    print(data)
data.iloc[:, 3:]

#%%
# Lets see where the results are different
if TERMINAL:
    print("\nAre they equal? Show differences below:")
    print(data.iloc[idxs1, :])
data.iloc[idxs1, 3:]


########################################################################
# d) Exploring the efficiency
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This code is used to compare whether the implementations are
# more or less efficient between them. Note that the methods
# have itself some limitations.

# Generate data
N = 10000000
choices = np.arange(2)
vector1 = np.random.choice(choices, size=N)
vector2 = np.random.choice(choices, size=N)

# Compute times
t1 = timer()
m1 = mutual_info_matrix_v1(x=vector1, y=vector2)
t2 = timer()
m2 = mutual_info_matrix_v2(x=vector1, y=vector2)
t3 = timer()
m3 = mutual_info_matrix_v3(x=vector1, y=vector2)
t4 = timer()

# Display
print("\n" + "="*80 + "\nExample 4\n" + "="*80)
print("Are the results equal? %s" % np.array_equal(m1, m2))
print("time v1: %.5f" % (t2-t1))
print("time v2: %.5f" % (t3-t2))
print("time v3: %.5f" % (t4-t3))


########################################################################
# e) Edge scenarios
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# There are some edge scenarios which we might have or might have not
# considered yet. We are including some of them here for future reference
# and some interesting questions below.
#
#   - What is the CRI range? (-0.7, 0.7)
#   - Should we normalize this value? [-1, 1]? [0, 1]?
#   - How to compute CRI if we have three outcomes R, S and I?

print("\n" + "="*80 + "\nExample 5\n" + "="*80)

# Create cases
data = [
    (['R', 'R', 'R', 'R'], ['R', 'R', 'R', 'R']),
    (['R', 'R', 'R', 'R'], ['S', 'S', 'S', 'S']),
    (['R', 'R', 'S', 'S'], ['R', 'R', 'S', 'S']),
    (['R', 'R', 'S', 'S'], ['S', 'S', 'R', 'R']),
    (['R', 'I', 'S', 'S'], ['R', 'I', 'S', 'S'])
]

# Results
cumu = []

# Loop
for i, (x, y) in enumerate(data):

    # Compute mutual information scores
    mis = mutual_info_score(x, y)
    misa = adjusted_mutual_info_score(x, y)
    misn = normalized_mutual_info_score(x, y)

    # Compute mutual information matrix
    m = mutual_info_matrix_v1(x=x, y=y)

    # Compute collateral resistance index
    try:
        cri = collateral_resistance_index(m)
    except Exception as e:
        print(e)
        cri = None

    # Append
    cumu.append([x, y, mis, misa, misn, cri])

    # Show
    print("\n%s. Contingency matrix:" % i)
    print(m)


# Create the dataframe
df = pd.DataFrame(cumu,
    columns=['x', 'y', 'mis', 'misa', 'misn', 'cri'])

#%%
# Lets see the summary of edge cases
if TERMINAL:
    print("\nSummary:")
    print(df)
df


########################################################################
# f) For continuous variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# There are several approaches, one of them is just binning. For more
# information just check online, there are many good resources and or
# implementations that might be found out there.