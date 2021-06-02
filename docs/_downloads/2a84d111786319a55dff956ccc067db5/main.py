"""
Main
=============

Example
"""
# Libraries
import json
import numpy as np

# Jsonpickle
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

# Configure jsonpickle
jsonpickle.set_preferred_backend('json')
jsonpickle_numpy.register_handlers()


# -------------------
# Main
# -------------------
"""
In order to test whether the jsonpickle encode method saves
the estimators so they can be retrieved no matter the numpy
version. Note that previously using pickle old estimators
could not be loaded because of numpy issues.

At the moment I have tested using the system (no virtual
environment) using the following...

    $ py -m pip install numpy==1.18
    $ py main.py
    $ py -m pip install numpy==1.19.1
    $ py main.py

And it works normally.

.. todo: Test it through a venvpy28-jsonpickle.
"""

# Path
path =  './estimator_00.json'

# Read
with open(path) as json_file:
    d = json.load(json_file)

# Show information
#print(d['iteration_00'])

# Show information
print("\nUsing numpy... <%s>" % np.version.version)

# Load
estimator = jsonpickle.decode(d['iteration_00'])

# Show
print("\nEstimator loaded. \n   %s" % estimator)
print("\nPredictions completed. \n   %s" % \
      estimator.predict_proba(np.arange(6).reshape(1,-1)))
