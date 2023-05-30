"""
02. ``dataprep``
======================================
"""
# Libraries
import pandas as pd

# Specific
from dataprep.eda import create_report
from sklearn.datasets import load_iris
from pathlib import Path

# Load data object
obj = load_iris(as_frame=True)

# Create report
profile = create_report(obj.data,
    title="Pandas Profiling Report")

# Save to file
Path('./outputs').mkdir(parents=True, exist_ok=True)
profile.save("./outputs/profile02-report.html")

# Show report in the browser
#profile.show_browser()