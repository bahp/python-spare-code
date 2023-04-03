import xgboost
import shap

X, y = shap.datasets.adult()
model = xgboost.XGBClassifier().fit(X, y)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.plots.beeswarm(shap_values, max_display=12, order=shap.Explanation.abs.mean(0))