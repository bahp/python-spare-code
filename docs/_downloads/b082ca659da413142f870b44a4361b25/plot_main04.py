"""
04. Basic example
=================

"""
# coding: utf-8

# In[1]:

### using XGBoost model with SHAP

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

import shap

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

shap.initjs()


# In[2]:

### make data
X, y = make_regression(n_samples=100, n_features=5,
    n_informative=3, random_state=0, noise=4.0,
    bias=10.0)
feature_names = ["x" + str(i+1) for i in range(0,5)]
data = pd.DataFrame(X, columns=feature_names)
data["target"] = y


# In[3]:

X_train, X_test, y_train, y_test = \
    train_test_split(data[feature_names], ## predictors only
                     data.target,
                     test_size=0.30,
                     random_state=0)


# In[4]:

### create and fit model
estimator = xgb.XGBRegressor()
estimator.fit(X_train, y_train)


# In[5]:

## kernel shap sends data as numpy array which has no column names, so we fix it
def xgb_predict(data_asarray):
    data_asframe =  pd.DataFrame(data_asarray, columns=feature_names)
    return estimator.predict(data_asframe)


# In[6]:

#### Kernel SHAP
X_summary = shap.kmeans(X_train, 10)
shap_kernel_explainer = shap.KernelExplainer(xgb_predict, X_summary)


# In[7]:

## shapely values with kernel SHAP
shap_values_single = shap_kernel_explainer.shap_values(X_test.iloc[[5]])
shap.force_plot(shap_kernel_explainer.expected_value, shap_values_single, X_test.iloc[[5]])



# In[9]:

#### Tree SHAP
shap_tree_explainer = shap.TreeExplainer(estimator)


# In[10]:

# Deprecated error
## shapely values with Tree SHAP
#shap_values_single = shap_tree_explainer.shap_values(X_test.iloc[[5]])
#shap.force_plot(shap_tree_explainer.expected_value, shap_values_single, X_test.iloc[[5]])

plt.show()