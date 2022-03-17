"""
Plot Tree
============

This example shows how to plot a Decision Tree

.. warning:: Broken because the library graphviz is missing
             and it gives an error when trying to install it.
"""

# Generic
import pybaobabdt
import pandas as pd

# Specfic
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier

# Load data
data = arff.loadarff('winequality-red.arff')
df   = pd.DataFrame(data[0])

# Define X and y
y = list(df['class'])
features = list(df.columns)
features.remove('class')
X = df.loc[:, features]

# Create classifier.
clf = DecisionTreeClassifier().fit(X,y)

# Draw tree
ax = pybaobabdt.drawTree(clf, size=10, dpi=72, features=features)

# Sve figure
#ax.get_figure().savefig('tree.png',
#   format='png', dpi=300, transparent=True)