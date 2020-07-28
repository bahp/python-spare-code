# Libraries
import pandas as pd

# Libraries specific
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer


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
                               inplace=True,
                               **kwargs):
        """
        """
        if requires_X:
            kwargs['X'] = self.get_X()
        if requires_y:
            kwargs['y'] = self.get_y()

        # Return
        if not inplace:
            return getattr(obj, func)(**kwargs)

        # Inplace
        self.dataframe[self.X_cols] = \
            getattr(obj, func)(**kwargs)



# --------------------------------------------------
# Main
# --------------------------------------------------
# Read data
dataframe = pd.read_csv('./dataset.csv')
dataframe = dataframe.reset_index()

# Define columns
X_columns = ['BIL', 'CRP', 'CRE']
y_columns = ['micro_confirmed']

# Create wrapper
wrapper = DataframeXYWrapper(dataframe=dataframe,
                             X_columns=X_columns,
                             y_columns=y_columns)
# Show
print(wrapper.get_X())

# Create scaler
std = StandardScaler()
mmx = MinMaxScaler()
rbs = RobustScaler()
nor = Normalizer()

# Apply transform
standard = wrapper.apply(std, 'fit_transform', requires_X=True, inplace=False)
minmax = wrapper.apply(mmx, 'fit_transform', requires_X=True, inplace=False)
robust = wrapper.apply(rbs, 'fit_transform', requires_X=True, inplace=False)
#normal = wrapper.apply(nor, 'fit_transform', requires_X=True, inplace=False)

# Show
print(standard)
print(minmax)
print(robust)
#print(normal)
print(wrapper.get_X_y())








