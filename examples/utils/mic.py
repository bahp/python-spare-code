"""
MIC package
---------------------------

v1 = [R, R, R, R]
v2 = [R, R, R, R]


"""
# Libraries
import numpy as np
import pandas as pd

from scipy.stats.contingency import crosstab
from sklearn.metrics import mutual_info_score


def mutual_info_matrix_v3(x=None, y=None, ct=None):
    """Compute the component information score.

    .. note: Might be inefficient but good for testing.

    .. note: In order to be able to compute the mutual
             information score it is necessary to have
             variation within the variable. Thus, if
             there is only one class, should we return
             a result or a warning?

    Parameters
    ----------
    x: list
        List with the classes
    y: list
        List with the classes

    Returns
    -------
    """

    def _check_nparray(obj, param_name):
        if obj is not None:
            if isinstance(obj, np.ndarray):
                return obj
            elif isinstance(obj, pd.Series):
                return obj.to_numpy()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_numpy()
            elif isinstance(obj, list):
                return np.array(obj)
            else:
                raise ValueError("""
                       The input parameter '{0}' is of type '{1} which is 
                       not supported. Please ensure it is a np.ndarray."""
                                 .format(param_name, type(obj)))

        # Ensure they are all np arrays

    x = _check_nparray(x, 'x')
    y = _check_nparray(y, 'y')
    ct = _check_nparray(ct, 'ct')

    # Compute contingency
    if ct is None:
        c = crosstab(x,y)
        if isinstance(c, tuple):
            ct = c[-1]   # older scipy
        else:
            ct = c.count # newer scipy

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

    # Return
    return m


def mutual_info_matrix_v2(x=None, y=None, ct=None):
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

    def _check_nparray(obj, param_name):
        if obj is not None:
            if isinstance(obj, np.ndarray):
                return obj
            elif isinstance(obj, pd.Series):
                return obj.to_numpy()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_numpy()
            elif isinstance(obj, list):
                return np.array(obj)
            else:
                raise ValueError("""
                       The input parameter '{0}' is of type '{1} which is 
                       not supported. Please ensure it is a np.ndarray."""
                                 .format(param_name, type(obj)))

        # Ensure they are all np arrays

    x = _check_nparray(x, 'x')
    y = _check_nparray(y, 'y')
    ct = _check_nparray(ct, 'ct')

    # Compute contingency
    if ct is None:
        c = crosstab(x, y)
        if isinstance(c, tuple):
            ct = c[-1]  # older scipy
        else:
            ct = c.count  # newer scipy

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

    # Return
    return m


def mutual_info_matrix_v1(x=None, y=None, *, ct=None):
    """Computes the mutual information matrix

    The component information is calculated as below where X/Y
    denotes a state (e.g. RR).

        C(X/Y) = P(XY) * log(P(XY) / P(X)*P(Y))
               = P(XY) * [log(P(XY)) - log(P(X)*P(Y))]
               = P(XY) * [log(P(XY)) - (log(P(X)) + log(P(Y)))]
               = P(XY) * [log(P(XY)) - log(P(X)) - log(P(Y))]

    .. note:: It is inspired by the code from sklearn.metrics.mutual_info_score.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    from math import log
    from scipy import sparse as sp
    from sklearn.metrics.cluster._supervised import check_clusterings
    from sklearn.metrics.cluster._supervised import check_array
    from sklearn.metrics.cluster._supervised import contingency_matrix

    labels_true, labels_pred, contingency = x, y, ct

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

    mi = (
        contingency_nm * (log_contingency_nm - np.log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)

    #return np.clip(mi.sum(), 0.0, None), mi
    try:
        return np.array(mi).reshape(contingency.shape).T
    except:
        return mi
        #return mutual_info_matrix_v3(x=x, y=y, ct=ct)