# Libraries
import pandas as pd

# Libraries specific
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#

_DEFAULT_SPLITTERS = {
  'skfold10': StratifiedKFold(n_splits=10, shuffle=True),
  'skfold5': StratifiedKFold(n_splits=5, shuffle=True),
  'skfold2': StratifiedKFold(n_splits=2, shuffle=True),
}

def skfold_acronym_to_instance(acronym):
    """
    """
    pass


def split_dataframe_hos_cvs(dataframe, **kwargs):
    """This method labels the dataframe hos and cvs sets.

    Parameters
    ----------

    Returns
    -------
    pd.DataFrame :
        The outcome is the same dataframe with an additional column
        <set> with the values cvs (cross-validation set) and hos
        (hold-out set).
    """
    # Split in hos and training sets
    cvs, hos = train_test_split(dataframe.index.to_numpy(), **kwargs)

    # Fill dataset
    dataframe['sets'] = None
    dataframe.loc[cvs, 'sets'] = 'cvs'
    dataframe.loc[hos, 'sets'] = 'hos'

    # Return
    return dataframe


def split_dataframe_cvs_folds(dataframe, splitter, label):
    """This method...
    """

    if isinstance(splitter, str):
        splitter = _DEFAULT_SPLITTERS[splitter]

    # Define X and y
    X = dataframe[dataframe.sets=='cvs'].index.to_numpy()
    y = dataframe[dataframe.sets=='cvs'][label]

    # Splits
    splits = splitter.split(X, y)

    # Loop and add
    for i, (train, test) in enumerate(splits):
        dataframe['set_iter_{0}'.format(i)] = None
        dataframe.loc[train, 'set_iter_{0}'.format(i)] = 'train'
        dataframe.loc[test, 'set_iter_{0}'.format(i)] = 'test'

    # Return
    return dataframe

def split_dataframe_completeness(dataframe):
    pass


# --------------------------------------------------
# Main
# --------------------------------------------------
# Read data
dataframe = pd.read_csv('./dataset.csv')
dataframe = dataframe.reset_index()

# Show
print(dataframe)

# Split in Hos and CVS sets
dataframe = split_dataframe_hos_cvs(dataframe)
dataframe = split_dataframe_cvs_folds(dataframe, splitter='skfold10',
                                                 label='micro_confirmed')

# Show
print(dataframe)






class DataframeHOSCSVSplitter():
    """

    """
    col_name = 'sets'
    cvs_name = 'CSV'
    hos_name = 'HOS'

    def __init__(self, col_name=None,
                       cvs_name=None,
                       hos_name=None):
        """

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
        """

        """
        # Split
        cvs, hos = train_test_split(dataframe.index.to_numpy(), **kwargs)

        # Fill dataset
        dataframe[self.col_name] = None
        dataframe.loc[cvs, self.col_name] = self.cvs_name
        dataframe.loc[hos, self.col_name] = self.hos_name

        # Return
        return dataframe
