"""
Shap - Main 07
==============

"""
# Libraries
import xgboost
import shap
import matplotlib.pyplot as plt

# Load shap dataset
X, y = shap.datasets.adult()

# Train model
model = xgboost.XGBClassifier().fit(X, y)

# Create shap explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)


# Create beeswarm plot using explainer
shap.plots.beeswarm(shap_values,
    max_display=12,
    order=shap.Explanation.abs.mean(0))

# Adjust
plt.tight_layout()