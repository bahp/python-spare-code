"""
02. Data splitters
==================

Useful methods to split data such as (i) hold out and
csv sets or (ii) cross validation folds.

.. warning:: Not completed!

"""
# Libraries
import pandas as pd
import numpy as np

# Libraries specific
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False

def split_dataframe_hos_cvs(dataframe,  **kwargs):
    """This method labels the dataframe hos and cvs sets.

    Parameters
    ----------
    dataframe: np.array or pd.DataFrame
        The data to be divided into HOS/CVS.

    Returns
    -------
    np.array:
        The outcome is a numpy array with rows labelled as
        cvs (cross-validation set) and hos (hold-out set).
        :param data:
        :param inplace:
    """
    # Check it is a dataframe
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError

    # Length
    n = dataframe.shape[0]

    # Split in hos and training sets
    cvs, hos = train_test_split(np.arange(n), **kwargs)

    # Create result
    empty = np.array([None]*n)
    empty[cvs] = 'cvs'
    empty[hos] = 'hos'

    # Include
    dataframe['sets'] = empty

    # Return
    return dataframe


def split_dataframe_cvs_folds(dataframe, splitter,
            selected_rows=None, **kwargs):
    """This method labels the different folds.

        .. note:

    Parameters
    ----------
    dataframe: np.array or pd.DataFrame
        The data to be divided into folds.

    splitter: str or splitter
        The splitter which can be an str or an splitter from the
        sklearn library which implementeds the method split.

    selected_rows: array of bools.
        The rows to be considered to create the folds. Note that if
        y is passed (for stratified cross validation) y will also be
        filtered by these rows.

    kwargs:

    Returns
    -------
    pd.DataFrame:
        The outcome is the same dataframe with an additional column
        <set> with the values cvs (cross-validation set) and hos
        (hold-out set).
    """
    # Check it is a dataframe
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError

    # Get splitter from string
    if isinstance(splitter, str):
        splitter = _DEFAULT_SPLITTERS[splitter]

    # Define X and y
    #X = dataframe[dataframe.sets == 'cvs'].index.to_numpy()
    #y = dataframe[dataframe.sets == 'cvs'][label]

    # Shape
    r, c = dataframe.shape

    # No rows selected (all by default)
    if selected_rows is None:
        selected_rows = np.full(r, True, dtype=bool)

    # Select rows from y
    if 'y' in kwargs:
        if kwargs['y'] is not None:
            kwargs['y'] = kwargs['y'][selected_rows]

    # Create indexes to use for splitting
    idxs = np.arange(r)[selected_rows].reshape(-1, 1)

    # Get splits of idxs
    splits = splitter.split(idxs, **kwargs)

    # Loop and add
    for i, (train, test) in enumerate(splits):
        dataframe['split_{0}'.format(i)] = None
        dataframe.loc[idxs[train].flatten(), 'split_{0}'.format(i)] = 'train'
        dataframe.loc[idxs[test].flatten(), 'split_{0}'.format(i)] = 'test'

    # Return
    return dataframe


def split_dataframe_completeness(dataframe):
    pass



class DataframeHOSCSVSplitter():
    """
    """
    col_name = 'sets'
    cvs_name = 'CVS'
    hos_name = 'HOS'

    def __init__(self, col_name=None,
                       cvs_name=None,
                       hos_name=None):
        """Constructor

        :param col_name:
        :param cvs_name:
        :param hos_name:
        """
        if col_name is not None:
            self.col_name = col_name
        if cvs_name is not None:
            self.cvs_name = cvs_name
        if hos_name is not None:
            self.hos_name = hos_name

    def split(self, dataframe, **kwargs):
        """Splits the dataframe...
        """
        # Split
        cvs, hos = train_test_split(dataframe.index.to_numpy(), **kwargs)

        # Fill dataset
        dataframe[self.col_name] = None
        dataframe.loc[cvs, self.col_name] = self.cvs_name
        dataframe.loc[hos, self.col_name] = self.hos_name

        # Return
        return dataframe



# Default splliters.
_DEFAULT_SPLITTERS = {
    'skfold10': StratifiedKFold(n_splits=10, shuffle=True),
    'skfold5': StratifiedKFold(n_splits=5, shuffle=True),
    'skfold2': StratifiedKFold(n_splits=2, shuffle=True),
}


# --------------------------------------------------
# Main
# --------------------------------------------------
# Libraries
from sklearn.datasets import load_iris

# Load data
bunch = load_iris(as_frame=True)

# Dataframe
dataframe = bunch.data

# %%
# Lets see the dataset

if TERMINAL:
    print("\nData")
    print(dataframe)
dataframe

# %%
# Lets split in HOS and CVS

# Split in HOS and CVS sets
df = split_dataframe_hos_cvs(dataframe)

if TERMINAL:
    print("\nData")
    print(df)
df

# %%
# Lets split the CSV in various folds

# Split in folds
df = split_dataframe_cvs_folds(dataframe,
    splitter='skfold5', y=bunch.target,
    selected_rows=(dataframe.sets == 'cvs'))

if TERMINAL:
    print("\nData")
    print(df)
df



# %%
# Lets split in HOS and CVS using the class
#
# .. note:: This might not be working properly!

# Divide in HOS and CSV.
df = DataframeHOSCSVSplitter().split(dataframe)

if TERMINAL:
    print("\nData")
    print(df)
df
