"""
05. IQRFilter
=============

Example of implementing an IQR filter.

"""

# Import from future.
from __future__ import division

# Libraries
import numpy as np

# --------------------------------------------------------------------------
#                       Inter-Quantile Range filter
# --------------------------------------------------------------------------
class IQRFilter():
    """This filter set those cells which lie outside of the
    interquantile range rule as np.nan. Ir performs iqr
    filtering for a single data matrix or a matrix with the
    corresponding classes. The latter performs the filtering
    for each class independently.

    .. todo: Return indicator with values set as nan.
    .. note: The code could be simplified.
    .. note: The coud could check input classes and raise error.

    """

    def __init__(self, iqrrange=[25, 75], coefficient=1.5):
        """The constructor"""
        self.iqrrange = iqrrange
        self.coefficient = coefficient
        self.lower_coefs = None
        self.upper_coefs = None

    def __repr__(self):
        """
        """
        return "IQRFilter(iqrrange=%s, coefficient=%s)" % \
               (self.iqrrange, self.coefficient)

    # --------------------------
    #   generic methods
    # --------------------------
    def _fit(self, X):
        """This method computes the lower and upper percentiles
        """
        # Compute lower and uper quartiles
        lower_quartiles, upper_quartiles = \
            np.nanpercentile(X, self.iqrrange, axis=0)

        # Compute the interquantile range
        iqrs = (upper_quartiles - lower_quartiles) * self.coefficient

        # Set parameters
        return lower_quartiles - iqrs, upper_quartiles + iqrs

    # --------------------------
    #   single class methods
    # --------------------------
    def _fit_s(self, X):
        """This method fits single category.

         Parameters
         ----------
         X :

         Returns
         -------
         IQRFIlter instance
        """
        # Create the array coefficients
        self.lower_coefs, self.upper_coefs = self._fit(X)

        # Format to array
        self.lower_coefs = self.lower_coefs.reshape(1, -1)
        self.upper_coefs = self.upper_coefs.reshape(1, -1)

        # Return
        return self

    def _filter_s(self, X):
        """This method filters single category.

        Parameters
        ----------
        X :

        Returns
        -------
        np.ndarray
        """
        # Copy X
        F = np.copy(X)

        # Indexes
        is_lower = F < self.lower_coefs[0, :]
        is_upper = F > self.upper_coefs[0, :]

        # Filter
        F[is_lower | is_upper] = np.nan

        # Return
        return F

    # ----------------------------
    #   multiple class methods
    # ----------------------------
    def _fit_m(self, X, y):
        """This method fits multiple category

        Note: the attribute _classes is a dictionary in which the key is
              the ategory identifier and the value is the index within
              the lower_coefs and upper_coefs.

        Parameters
        ----------
        X :
        y :

        Returns
        -------
        IQRFIlter instance
        """
        # Create matrices with coefficients for each class
        self.lower_coefs = [self._fit(X[y == c])[0] for c in np.unique(y)]
        self.upper_coefs = [self._fit(X[y == c])[1] for c in np.unique(y)]

        # Format to array
        self.lower_coefs = np.array(self.lower_coefs)
        self.upper_coefs = np.array(self.upper_coefs)

        # Set classes
        self._classes = {c: idx for idx, c in enumerate(np.unique(y))}

        # Return
        return self

    def _filter_m(self, X, y):
        """This method filters multiple category.


        Parameters
        ----------
        X :
        y :

        Returns
        -------
        np.ndarray
        """
        # Copy matrix
        F = np.copy(X).astype(float)

        # For each category
        for category, index in self._classes.items():
            # Indexes
            is_y = np.repeat((y == category).reshape(-1, 1), X.shape[1], 1)
            is_lower = F < self.lower_coefs[index, :]
            is_upper = F > self.upper_coefs[index, :]

            # Filter
            F[is_y & (is_lower | is_upper)] = np.nan

        # Return
        return F

    # -------------------------
    #   caller methods
    # -------------------------
    def fit(self, X, y=None):
        """This method fits the filter to the data.

        Parameters
        ----------
        X :
        y :

        Returns
        -------
        self
        """
        # Fit filter
        if y is None:
            self._fit_s(X)
        else:
            self._fit_m(X, y)
        # Return
        return self

    def filter(self, X, y=None):
        """This method filters the input

        Parameters
        ----------
        X :

        Returns
        -------
        np.ndarray
        """
        # The object has not been previously fitted
        if self.lower_coefs is None or self.upper_coefs is None:
            raise TypeError("The instance IQRFilter has not been fitted.")

        # The instance has been fitted with classes
        if hasattr(self, '_classes') and y is None:
            raise TypeError("The instance IQRFilter has been fitted with "
                            "several categories (%s). As such, the y "
                            "parameter is required to identify the "
                            "categories." % self._classes.keys())

        # Verify that all classes are included
        if hasattr(self, '_classes'):
            y_classes = set(np.unique(y))
            f_classes = set(self._classes.keys())
            if bool(y_classes - f_classes):
                raise TypeError("There are categories in the inputed y (%s) which "
                                "were not seen during the fiting process (%s). As "
                                "such the data cannot be filtered." %
                                (y_classes, f_classes))

        # Filter
        if y is None:
            return self._filter_s(X)
        # Multiple category
        return self._filter_m(X, y)

    def fit_filter(self, X, y=None):
        """This method fits and filters.
        """
        # Fit
        self.fit(X, y)
        # Filter
        if y is None:
            return self.filter(X)
        # Return
        return self.filter(X, y), y




if __name__ == '__main__':

    # Import
    import numpy as np
    import warnings
    import matplotlib as mpl

    # Import specific
    from sklearn.datasets import make_classification

    # ------------------------------------
    # basic configuration
    # ------------------------------------
    # Ignore all the warnings
    warnings.simplefilter('ignore')

    # Set matplotlib
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    mpl.rcParams['axes.titlesize'] = 11
    mpl.rcParams['legend.fontsize'] = 9

    # Set print options
    np.set_printoptions(precision=2)

    # ------------------------------------
    # create data
    # ------------------------------------
    # Create feature data
    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 1, 2, 3],
                     [1, 2, 3, 4],
                     [1, 2, 3, 3],
                     [3, 7, 3, 4],
                     [1, 2, 3, 3],
                     [3, 7, 3, 4],
                     [1, 2, 3, 4],
                     [3, 6, 3, 4],
                     [2, 2, -55, 55]])

    # Create categories
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # --------------
    # IQR filtering
    # --------------
    # Create filter object
    iqr = IQRFilter(iqrrange=[25, 75], coefficient=1.5)

    # Fit and filter
    X, y = iqr.fit_filter(data, y)

    # Show
    print(X)
    print(y)