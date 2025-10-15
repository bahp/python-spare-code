"""
01. Explainer Dash
====================

This example runs an app which an interactive
user interface to explore the importance of
the features using SHAP values.

.. note:: https://github.com/oegedijk/explainerdashboard

.. note:: See https://titanicexplainer.herokuapp.com/

.. note:: Requires plotly<6

"""
# Libraries
import sys
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer
from explainerdashboard import ExplainerDashboard
from explainerdashboard.datasets import titanic_survive
from explainerdashboard.datasets import titanic_names
from explainerdashboard.datasets import feature_descriptions


try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False



def check_version(name):
    """"""
    if sys.version_info >= (3, 8):
        # Since python 3.8
        from importlib.metadata import version, PackageNotFoundError
    else:
        # For older Python versions, you need to install the backport
        from importlib_metadata import version, PackageNotFoundError

    try:
        pkg_version = version(name)
        print(f"The version of '{name}' is: {pkg_version}")
        return pkg_version
    except PackageNotFoundError:
        print("One of the packages is not installed.")
        return None




if __name__ == '__main__':

    if check_version('plotly') < '6':

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
        if TERMINAL:
            ExplainerDashboard(explainer).flask_server()