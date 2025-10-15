"""
02. Generate an EDA Report with Dataprep
========================================

This script demonstrates how to quickly generate a comprehensive Exploratory
Data Analysis (EDA) report using the `dataprep` library. It loads the classic
Iris dataset, creates an interactive report, and saves it as an HTML file.

.. note:: Dataprep Installation on Windows

   The installation of the `dataprep` library may fail on a standard Windows
   environment. This is because some of its dependencies (like `levenshtein`
   and `regex`) need to be compiled from C++ source code during installation.

   To resolve this, you must have the **Microsoft C++ Build Tools** installed on
   your system.

   1.  **Download the installer:**
       https://visualstudio.microsoft.com/visual-cpp-build-tools/

   2.  **Run the installer** and select the **"Desktop development with C++"**
       workload.

   3.  Once installed, restart your terminal and try the installation again:
       `pip install dataprep`
"""

"""
.. note:: Lots of issues with conflicting libraries. But
          should work in its own virtual environment.

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
"""