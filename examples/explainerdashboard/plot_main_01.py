"""
Explainer Dash
----------------

This example runs an app which an interactive
user interface to explore the importance of
the features using SHAP values.

.. note:: https://github.com/oegedijk/explainerdashboard

"""
# Libraries
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer
from explainerdashboard import ExplainerDashboard
from explainerdashboard.datasets import titanic_survive
from explainerdashboard.datasets import feature_descriptions

# Get data
X_train, y_train, X_test, y_test = titanic_survive()

# Create and fit model
model = RandomForestClassifier(n_estimators=50, max_depth=10) \
    .fit(X_train, y_train)

# Configure explainer
explainer = ClassifierExplainer(model, X_test, y_test,
       cats=['Sex', 'Deck', 'Embarked'],
       descriptions=feature_descriptions,
       labels=['Not survived', 'Survived'])

# Run
#ExplainerDashboard(explainer).run()