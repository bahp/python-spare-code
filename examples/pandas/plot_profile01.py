"""
01. Generate an EDA Report with Pandas Profiling
================================================

This script demonstrates how to generate a detailed Exploratory Data Analysis
(EDA) report with a single line of code using the `pandas-profiling` library.
It loads the standard Iris dataset, creates a comprehensive and interactive
data profile, and saves the final report as a self-contained HTML file.

.. note::
   The `pandas-profiling` library has been renamed to `ydata-profiling`.
   While the old import may still work for backward compatibility, it is
   recommended to install and import `ydata-profiling` in new projects.
"""

# Libraries
import pandas as pd

# Specific
from pandas_profiling import ProfileReport
from sklearn.datasets import load_iris
from pathlib import Path

# Load data object
obj = load_iris(as_frame=True)

# Create report
profile = ProfileReport(obj.data,
    title="Pandas Profiling Report",
    explorative=True)

# Save to file
Path('./outputs').mkdir(parents=True, exist_ok=True)
profile.to_file("./outputs/profile01-report.html")

# %%
# Show

profile