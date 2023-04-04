"""
Pandas Profile - Main 01
========================
"""
# Libraries
import pandas as pd

# Specific
from pandas_profiling import ProfileReport

# path
path = './data/dataset.csv'

# Load csv
data = pd.read_csv(path)

# Show
print(data)

# Create report
profile = ProfileReport(data,
    title="Pandas Profiling Report",
    explorative=True)

# Save to file
profile.to_file("report.html")