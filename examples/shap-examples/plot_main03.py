"""
Shap - Main 03
==============

"""
# Generic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Xgboost
from xgboost import XGBClassifier

# ----------------------------------------
# Load data
# ----------------------------------------
# Seed
seed = 0

# Load dataset
bunch = load_iris()
bunch = load_breast_cancer()
features = list(bunch['feature_names'])

# Create DataFrame
data = pd.DataFrame(data=np.c_[bunch['data'], bunch['target']],
                    columns=features + ['target'])

# Create X, y
X = data[bunch['feature_names']]
y = data['target']

# Filter
X = X.iloc[:500, :3]
y = y.iloc[:500]


# Split dataset
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state=seed)


# ----------------------------------------
# Classifiers
# ----------------------------------------
# Train classifier
gnb = GaussianNB()
llr = LogisticRegression()
dtc = DecisionTreeClassifier(random_state=seed)
rfc = RandomForestClassifier(random_state=seed)
xgb = XGBClassifier(
    min_child_weight=0.005,
    eta= 0.05, gamma= 0.2,
    max_depth= 4,
    n_estimators= 100)

# Select one
clf = rfc

# Fit
clf.fit(X_train, y_train)

# ----------------------------------------
# Find shap values
# ----------------------------------------
# Import
import shap

# Initialise
shap.initjs()

# Get generic explainer
explainer = shap.Explainer(clf, X_train)

# Show kernel type
print("\nKernel type: %s" % type(explainer))

# Variables
#rows = X_train.iloc[5, :]
#shap = explainer.shap_values(row)

# Get shap values
shap_values = explainer.shap_values(X_train.iloc[5, :])
#shap_values = explainer.shap_values(X_test.iloc[5, :])

print(shap_values)

# Force plot
# .. note: not working!
plot_force = shap.plots.force(explainer.expected_value[1],
    shap_values[1], X_train.iloc[5, :],
    matplotlib=True, show=True)

plt.tight_layout()
plt.show()

"""
import sys
sys.exit()

# ----------------------------------------
# Visualize
# ----------------------------------------
# Initialise
shap.initjs()
"""
"""
# Dependence plot
shap.dependence_plot(0, shap_values,
    X_train, interaction_index=None, dot_size=5,
    alpha=0.5, color='#3F75BC', show=False)
plt.tight_layout()
"""

"""
print(explainer.shap_values(X_train))

# Summary plot
plot_summary = shap.summary_plot( \
    explainer.shap_values(X_train),
    X_train, cmap='viridis',
    show=False)

plt.tight_layout()
plt.show()

print(plot_summary)


import seaborn as sns
sv = explainer.shap_values(X_train)
sv = pd.DataFrame(sv, columns=X.columns)
sv = sv.stack().reset_index()
sv['val'] = X_train.stack().reset_index()[0]

#import plotly.express as px

#f = px.strip(data_frame=sv, x=0, y='level_1', color='val')
#f.show()

print(sv)
#sns.swarmplot(data=sv, x=0, y='level_1', color='viridis', palette='viridis')
#sns.stripplot(data=sv, x=0, y='level_1', color='viridis', palette='viridis')
#plt.show()
import sys
sys.exit()
#sns.swarmplot(x=)

import sys
sys.exit()

#html = f"<head>{shap.getjs()}</head><body>"
# Bee swarm
# .. note: unexpected algorithm matplotlib!
# .. note: does not return an object!
plot_bee = shap.plots.beeswarm(shap_values, show=False)

# Sow
print("\nBEE")
print(plot_bee)

#print(f)
# Waterfall
# .. note: not working!
#shap.plots.waterfall(shap_values[0], max_display=14)

# Force plot
# .. note: not working!
plot_force = shap.plots.force(explainer.expected_value,
    explainer.shap_values(X_train), X_train,
    matplotlib=False, show=False)

# Show
print("\nFORCE:")
print(plot_force)
print(plot_force.html())
print(shap.save_html('e.html', plot_force))
"""
