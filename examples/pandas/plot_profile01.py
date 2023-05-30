"""
01. DataFrame description (``pandas_profiling``)
================================================
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