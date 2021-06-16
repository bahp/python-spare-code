"""
Author: Bernard Hernandez
Date: 11/09/2020
Description:

   In order to use high plot, first install the library
   and then run it using the epicimpoc fetcher as indicated
   below:

     $ python -m pip install hiplot
     $ hiplot fetcher.fetch_epicimpoc_experiment

   .. todo: Integrate both fetchers into one. Thus, the
            user can select the file:
            result.csv
            outcome.csv
            dataset.csv
"""
# Generic
import os
import hiplot as hip
import pandas as pd

# Pathlib
from pathlib import Path

# This is the default experiment path that will be used in the
# fetch method for epicimpoc experiments displayed below. Note
# that this might need to be redefined if running in another
# folder.
_DEFAULT_EXPERIMENT_PATH = \
    Path(os.path.abspath(__file__)).parent / 'datasets'

def fetch_epicimpoc_dataset(uri):
    """

    :param uri:
    :return:
    """
    # ---------------------------
    # Find csv file to load
    # ---------------------------
    # Only apply this fetcher if the URI starts with myxp://
    PREFIX = "myxp://"

    # Let other fetchers handle this one
    if not uri.startswith(PREFIX):
        raise hip.ExperimentFetcherDoesntApply()

    # Get pattern
    pattern = uri[len(PREFIX):]  # Remove the prefix
    if not pattern.endswith('csv'):
        pattern += '/dataset.csv'
    if not pattern.startswith('**'):
        pattern = '**/' + pattern

    # Find csv file.
    path = list(Path(_DEFAULT_EXPERIMENT_PATH).glob(pattern))

    # Raise errors
    # Since the file name is results.csv it is impossible
    # to have two files with the same name in the same
    # folder. Leave it just in case.
    if len(path) == 0:
        raise Exception("""No file found matching the search pattern '{0}'.""".format(pattern))
    if len(path) > 1:
        raise Exception("""Multiple files found matching the search pattern '{0}'.
            The files are:\n{1} """.format(pattern, path))

    # ---------------------------
    # Load and format data
    # ---------------------------
    features = sorted(set(['alp', 'alt', 'bil', 'cre', 'crp', 'wbc']))

    # Read csv
    data = pd.read_csv(path[0])
    data = data[['micro_confirmed'] + features]

    print(data)

    # ---------------------------
    # Create experiment
    # ---------------------------
    # Create experiment with data (from_csv with filepath too).
    exp = hip.Experiment.from_dataframe(data)

    """
    # Provide configuration for the parallel plot
    exp.display_data(hip.Displays.PARALLEL_PLOT).update({
        # 'hide': ['optionB'], # Hide from parallel plot
        'order': sorted(data.columns.values),
        'color_by': ['__gmean']  # Does not work
    })

    # Provide configuration for the table with all the rows
    exp.display_data(hip.Displays.TABLE).update({
        'hide': ['from_uid'],  # Hide columns
        'order_by': [['__gmean', 'desc']],  # Order
    })

    # Force ranges of scores (and therefore colors) between 0 and 1.
    # exp.parameters_definition["_gmean"].force_range(0, 1)
    # exp.parameters_definition["_sens"].force_range(0, 1)
    # exp.parameters_definition["_spec"].force_range(0, 1)
    # exp.parameters_definition["_aucroc"].force_range(0, 1)
    """

    # Return
    return exp


def fetch_epicimpoc_experiment(uri):
    """Fetcher for EPiC IMPOC experiments created with pySML library.

    The uri is expect to be of the form suggested below. This fetch
    method looks for outputs within the 'examples' folder in the
    current python-epicimpoc-inference.

           myxp://20200820-145219-completed/simp-std-smt-dtc
           myxp://20200820-145219-completed/simp-std-smt-dtc/results.csv

    .. note: A 'results.csv' file needs to exist within the folder.
    .. note: Raises error if multiple files found.
    .. note: Raises error If no file found.

    """
    # ---------------------------
    # Find csv file to load
    # ---------------------------
    # Only apply this fetcher if the URI starts with myxp://
    PREFIX = "myxp://"

    # Let other fetchers handle this one
    if not uri.startswith(PREFIX):
        raise hip.ExperimentFetcherDoesntApply()

    # Get pattern
    pattern = uri[len(PREFIX):]  # Remove the prefix
    if not pattern.endswith('csv'):
        pattern += '/results.csv'
    if not pattern.startswith('**'):
        pattern = '**/' + pattern

    print(Path(_DEFAULT_EXPERIMENT_PATH))

    # Find csv file.
    path = list(Path(_DEFAULT_EXPERIMENT_PATH).glob(pattern))

    # Raise errors
    # Since the file name is results.csv it is impossible
    # to have two files with the same name in the same
    # folder. Leave it just in case.
    if len(path) == 0:
        raise Exception("""No file found matching the search pattern '{0}'.""".format(pattern))
    if len(path) > 1:
        raise Exception("""Multiple files found matching the search pattern '{0}'.
            The files are:\n{1} """.format(pattern, path))

    # ---------------------------
    # Load and format data
    # ---------------------------
    # Read csv
    data = pd.read_csv(path[0])

    # Helper method
    def contains(columns, name):
        for c in columns:
            if name in c:
                return True
        return False

    has_hos = contains(data.columns, '_hos_')
    has_test = contains(data.columns, '_test_')
    key = 'hos' if has_hos else 'test'

    # Rename columns
    keep = ['pipeline',
            'mean_%s_sens' % key,
            'mean_%s_spec' % key,
            'mean_%s_gmean' % key,
            'mean_%s_aucroc' % key]
    keep += [c for c in data.columns if c.startswith('param_')]  # keep params
    keep = [c for c in keep if c in data.columns]  # keep if exists

    # Filter
    data = data[keep]
    data = data.round(3)

    rename = {
        'mean_%s_sens' % key: '_sens',
        'mean_%s_spec' % key: '_spec',
        'mean_%s_gmean' % key: '__gmean',
        'mean_%s_aucroc' % key: '_aucroc'
    }

    # Remove param_svm/param_ann/... from names.
    data.columns = [c.split('__')[-1] for c in data.columns]
    # data.columns = [c.replace('mean','_mean') for c in data.columns] # issues gmean!

    data = data.rename(columns=rename)

    # Show data columns in server terminal
    print("Columns: %s" % str(data.columns.values))

    # ---------------------------
    # Create experiment
    # ---------------------------
    # Create experiment with data (from_csv with filepath too).
    exp = hip.Experiment.from_dataframe(data)

    # Provide configuration for the parallel plot
    exp.display_data(hip.Displays.PARALLEL_PLOT).update({
        # 'hide': ['optionB'], # Hide from parallel plot
        'order': sorted(data.columns.values),
        'color_by': ['__gmean']  # Does not work
    })

    # Provide configuration for the table with all the rows
    exp.display_data(hip.Displays.TABLE).update({
        'hide': ['from_uid'],  # Hide columns
        'order_by': [['__gmean', 'desc']],  # Order
    })

    # Force ranges of scores (and therefore colors) between 0 and 1.
    # exp.parameters_definition["_gmean"].force_range(0, 1)
    # exp.parameters_definition["_sens"].force_range(0, 1)
    # exp.parameters_definition["_spec"].force_range(0, 1)
    # exp.parameters_definition["_aucroc"].force_range(0, 1)

    # Return
    return exp
