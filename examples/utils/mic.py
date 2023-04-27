
# Libraries
import numpy as np
import pandas as pd

from scipy.stats.contingency import crosstab
from sklearn.metrics import mutual_info_score


def component_info_score_v3(x=None, y=None, ct=None):
    """Compute the component information score.

    .. note: Might be inefficient but good for testing.

    Parameters
    ----------
    x: list
        List with the classes
    y: list
        List with the classes

    Returns
    -------
    """
    # Compute contingency
    if ct is None:
        c = crosstab(x,y)
        ct = c.count

    # Variables
    n = ct.sum()
    pi = np.ravel(ct.sum(axis=1)) / n
    pj = np.ravel(ct.sum(axis=0)) / n

    # Create empty matrix
    m = np.empty(ct.shape)
    m[:] = np.nan

    # Fill with component information score
    with np.errstate(all='ignore'):
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                pxy = ct[i,j] / n
                m[i,j] = pxy * np.log(pxy / (pi[i] * pj[j]))

    # Fill with na (lim x->0 => 0)
    m[np.isnan(m)] = 0

    # Compute score
    score = (m[0, 0] + m[1, 1]) - (m[0, 1] + m[1, 0])

    # Return
    return score, m


def component_info_score_2d(x=None, y=None, ct=None):
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
    # Compute contingency
    if ct is None:
        c = crosstab(x,y)
        ct = c.count

    with np.errstate(divide='ignore'):
        # Variables
        n = ct.sum()
        pi = np.ravel(ct.sum(axis=1))
        pj = np.ravel(ct.sum(axis=0))

        # Compute matrix
        b = np.repeat(pi.reshape(-1,1), len(pi), axis=1)
        c = np.repeat(pj.reshape(1,-1), len(pj), axis=0)
        m = (ct/n) * np.log((ct/n) / ((b/n) * (c/n)))

    # Fill with na (lim x->0 => 0)
    m[np.isnan(m)] = 0

    # Compute score
    score = (m[0, 0] + m[1, 1]) - (m[0, 1] + m[1, 0])

    # Return
    return score, m