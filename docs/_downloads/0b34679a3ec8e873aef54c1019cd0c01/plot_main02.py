"""
02. Explainers Across ML Classifiers
=========================================================
This script serves as a practical guide to applying the SHAP
library across a diverse set of machine learning algorithms.
It highlights a critical concept in model interpretability:
different model architectures require specific types of SHAP
explainers for accurate and efficient computation. 🤖

The workflow includes:

    - **Training Various Models:** A suite of classifiers from scikit-learn
      and XGBoost are trained, including LogisticRegression, RandomForestClassifier,
      SVC, and XGBClassifier.
    - **Applying Appropriate Explainers:** The script demonstrates how to select
      and use different explainers, primarily contrasting the model-agnostic
      shap.KernelExplainer with the highly optimized explainer for tree-based
      models.
    - **Visual Comparison:** For each classifier, a SHAP summary plot is generated,
      allowing for a side-by-side comparison of feature importances as interpreted
      by each model.

This example is invaluable for understanding the practical nuances
of using SHAP and for choosing the correct approach to explain the
predictions of your specific machine learning model.

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
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

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
ann = MLPClassifier()
svm = SVC(probability=True)
etc = ExtraTreesClassifier()

# List
clfs = [gnb, llr, dtc, rfc, xgb, ann, svm, etc]
#clfs = [svm, dtc]

# Fit
for clf in clfs:
    clf.fit(X_train, y_train)

# ----------------------------------------
# Find shap values
# ----------------------------------------
# Possible explainers:
#    - shap.DeepExplainer
#    - shap.KernelExplainer
#    - shap.TreeExplainer
#    - shap.LinearExplainer
#    - shap.Exact
#    - shap.Explainer

# Import
import shap

# Initialise
shap.initjs()


def predict_proba(x):
    return clf.predict_proba(x)[:, 1]

# Loop
for clf in clfs:

    try:
        # Show classifier
        print("\n" + '-'*80)
        print("Classifier: %s" % clf)

        """
        # Create shap explainer
        if isinstance(clf,
            (DecisionTreeClassifier,
             ExtraTreesClassifier,
             XGBClassifier)):
            # Set Tree explainer
            explainer = shap.TreeExplainer(clf)
        elif isinstance(clf, LogisticRegression):
            # Masker
            masker = shap.maskers.Independent(X_train, max_samples=100)
            # Set Linear explainer
            #explainer = shap.LinearExplainer(predict_proba)#, masker)
            explainer = shap.Explainer(predict_proba, masker)
        elif isinstance(clf, int):
            # Set NN explainer
            explainer = shap.DeepExplainer(clf)
        else:
            # Works for [svc]
            # If too many examples (pass aux to explainer).
            aux = shap.sample(X_train, 100)
            # Set generic kernel explainer
            explainer = shap.KernelExplainer(predict_proba, aux)
        """

        # Sample to speed up processing.
        sample = shap.sample(X_train, 100)

        if isinstance(clf, XGBClassifier):
            # Works for [llr, dtc, etc, xgb]
            explainer = shap.Explainer(clf, sample)
        else:
            # Works for all but [xgb]
            explainer = shap.KernelExplainer(predict_proba, sample)

        # Show kernel type
        print("Kernel type: %s" % type(explainer))

        # Get shap values
        #shap_values = explainer(X)
        shap_values = explainer.shap_values(X_train)

        print(shap_values)


        # Show information
        print("base value: %s" % explainer.expected_value)
        #print("shap_values: %s" % str(shap_values.shape))

        # Summary plot
        plt.figure()
        plot_summary = shap.summary_plot(
            explainer.shap_values(X_train),
            X_train, cmap='viridis', show=False
        )

        # Format
        plt.title(clf.__class__.__name__)
        plt.tight_layout()

    except Exception as e:
        print("Error: %s" % e)

# Show
plt.show()
