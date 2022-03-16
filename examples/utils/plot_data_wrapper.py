"""
Data Wrapper
=============

.. warning:: Non sense.

"""

# Libraries
import pandas as pd

# Libraries specific
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer


class DataframeXYWrapper():

    def __init__(self, dataframe, X_columns, y_columns):
        """
        """
        self.dataframe = dataframe
        self.X_cols = X_columns
        self.y_cols = y_columns

    def get_X(self):
        """Return..."""
        return self.dataframe[self.X_cols]

    def get_y(self):
        """Return..."""
        return self.dataframe[self.y_cols]

    def get_X_y(self):
        """Return...
        """
        return self.dataframe[self.X_cols + self.y_cols]

    def apply(self, obj, func, requires_X=False,
                               requires_y=False,
                               argname_X='X',
                               argname_y='y',
                               inplace=True,
                               return_obj=True,
                               **kwargs):
        """
        """
        if requires_X:
            kwargs[argname_X] = self.get_X()
        if requires_y:
            kwargs[argname_y] = self.get_y()

        # Output
        out = getattr(obj, func)(**kwargs)

        # Inplace
        if inplace:
            self.dataframe[self.X_cols] = out

        # Return
        if return_obj:
            return obj, out
        return out



# --------------------------------------------------
# Main
# --------------------------------------------------
# Libraries
import numpy as np

N = 2000 # number of observations
F = 7    # number of features
P = 70   # percent of cells with nan

# Create dataset
X = np.random.randn(N, F)
y = (np.random.rand(N) > 0.1).astype(int)

# Include NaN
idxs = np.random.choice(X.size, int(N*F*P/100), replace=False)
X.ravel()[idxs] = np.nan

# Create dataframe
dataframe = pd.DataFrame(X)
dataframe['y'] = y

# Create wrapper
wrapper = DataframeXYWrapper(dataframe=dataframe,
                             X_columns=list(range(F)),
                             y_columns=['y'])
# Show
print("Features:")
print(wrapper.get_X())

# ------------------------------
# Imputers
# ------------------------------
# Loop
for name, imputer in [
            ('mean', SimpleImputer(strategy='mean')),
            ('median', SimpleImputer(strategy='median'))]:
    # Apply imputer
    obj, out = wrapper.apply(imputer, 'fit_transform',
            requires_X=True, inplace=False)
    # Display
    print("\n%s" % imputer)
    print(pd.DataFrame(out))


# ------------------------------
# Scalers
# ------------------------------
# Loop
for name, scaler in [('std', StandardScaler()),
                     ('mmx', MinMaxScaler()),
                     ('rbs', RobustScaler())]:
    # Apply scaler
    obj, out = wrapper.apply(scaler, 'fit_transform',
            requires_X=True, inplace=False)
    # Display
    print("\n%s" % scaler)
    print(pd.DataFrame(out))
