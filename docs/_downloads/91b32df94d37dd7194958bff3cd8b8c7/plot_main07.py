"""
07. SHAP Beeswarm Plot
=========================================================

This script provides a concise, fundamental example of how to
generate a SHAP beeswarm plot to visualize global feature
importance. 🤖 It trains an XGBoost classifier on the adult
census dataset, computes SHAP values, and then creates the
classic beeswarm summary plot, which displays the impact of
the most influential features on the model's output.

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