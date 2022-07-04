"""
Best pipeline
=============

Example
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score
from sklearn.model_selection._validation import _aggregate_score_dicts

from pySML2.settings import _DEFAULT_METRICS

# ---------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------
# The input features and label for the algorithm
features = sorted(['alb', 'alp', 'alt', 'baso', 'bil', 'cl', 'cre', 'crp', 'egfr',
                  'eos', 'k', 'ly',  'mcv', 'mono', 'mpv', 'nrbca', 'plt', 'rbc',
                  'rdw',  'urea', 'wbc'])

# The labels
labels = sorted(['micro_confirmed'])

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('tpot_data_train.csv', sep=',')
tpot_data.columns = [c.lower() for c in tpot_data.columns.values]
tpot_data = tpot_data[features + labels]
tpot_data = tpot_data.rename(columns={'micro_confirmed':'target'})

features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.739462953567469
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=69),
    ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.6000000000000001, min_samples_leaf=12, min_samples_split=14, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print(_DEFAULT_METRICS)


scorers, _ = _check_multimetric_scoring(exported_pipeline, scoring=_DEFAULT_METRICS)
scores = _score(exported_pipeline, testing_features, testing_target, scorers)
print(scores)
scores = _aggregate_score_dicts(scores)