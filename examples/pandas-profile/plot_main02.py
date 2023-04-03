"""
Pandas Profile - Main 02
========================
"""
# Libraries
import pandas as pd

# Specific
from dataprep.eda import create_report

# path
path = './data/dataset.csv'

# Load csv
data = pd.read_csv(path)

# Show
print(data)

# Create report
profile = create_report(data,
    title="Pandas Profiling Report")

# Save to file
profile.to_file("report.html")