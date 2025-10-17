"""
02. Basic TPOT example with manually created data
=================================================

This example demonstrates how to use TPOT with a synthetic dataset
created directly within the script, without loading external files.

.. note:: Dask needs bokeh>=1.3.0 for dashboard.
.. note:: Go to the HTTP server to see tasks (System tab)
"""

# Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
import os

# ---------------------------------------------
# Configuration
# ---------------------------------------------
# Create a directory to save the output file
output_dir = 'outcomes/main02'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ---------------------------------------------
# Create a synthetic dataset
# ---------------------------------------------
# We'll generate a binary classification problem with 500 samples and 25 features.
# - n_informative: number of features that are actually useful.
# - n_redundant: number of features that are linear combinations of informative features.
# - n_classes: number of distinct classes for the label.
# - random_state: ensures the data generation is reproducible.
X, y = make_classification(n_samples=500,
                           n_features=25,
                           n_informative=15,
                           n_redundant=5,
                           n_classes=2,
                           random_state=42)

# ---------------------------------------------
# Split data into training and testing sets
# ---------------------------------------------
# We'll use 75% of the data for training the TPOT pipeline
# and the remaining 25% for testing its final performance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.75,
    test_size=0.25,
    random_state=42
)

# ---------------------------------------------
# Automated Machine Learning Search with TPOT
# ---------------------------------------------
# Create the TPOT genetic search instance.
# - generations: Number of iterations to run the optimization process.
# - population_size: Number of machine learning pipelines to keep in each generation.
# - cv: The number of folds to use in cross-validation.
# - scoring: The metric to optimize for (Area Under the ROC Curve in this case).
# - verbosity: Set to 2 to see the progress of the search.
# - random_state: Ensures the TPOT search is reproducible.
tpot = TPOTClassifier(generations=5,
                      population_size=20,
                      cv=5,
                      #scoring='roc_auc', # not working... why?
                      #verbosity=2,       # not working... why?
                      random_state=42)

# Start the TPOT automated search process.
# TPOT will explore various models and preprocessing steps to find the best pipeline.
# Note: y_train is flattened with .ravel() to ensure it's a 1D array, a common requirement.
tpot.fit(X_train, y_train.ravel())

# ---------------------------------------------
# Evaluate and Export the Best Pipeline
# ---------------------------------------------
# Use the final fitted pipeline to make predictions on the unseen test set.
score = tpot.score(X_test, y_test.ravel())
print(f"\nTPOT pipeline's ROC AUC score on the test set: {score:.4f}")

# Export the Python code for the best pipeline found by TPOT.
# This file can be used to run the pipeline without needing to run the search again.
output_path = os.path.join(output_dir, 'tpot_manual_data_pipeline.py')
tpot.export(output_path)
print(f"The best pipeline has been exported to '{output_path}'")