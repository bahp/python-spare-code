"""
Generate data AST
-----------------

"""

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
CONFIGURATION = [
    {
        'specimen': 'BLDCUL',
        'microorganism': 'ORG1',
        'antmicrobial': 'ABX1',
        #'range': [2010, 2020],
        #'kwargs':
    },

]


def create_ast_from_values(y, ns=None):
    """Create AST records.

    .. note:: The values inputed must be between 0 and 100.

    Parameters
    ----------
    y: np.array
        The array with the resistance values, that is the proportion
        between resistant and isolates (R/R+S). All the values must
        be within the range [0, 100].

    Returns
    -------
    """

    # Create number of positive.
    if ns is None:
        ns = np.ones(shape=(1, y.shape[0])).astype(int) * 100
    if isinstance(ns, list):
        ns = np.array(ns)

    # Compute number of negatives.
    nr = ((ns*y) / (1-y)).round(decimals=0).astype(int)

    # Address extreme cases
    print(nr)
    print(ns)
    print(sari)

    import sys
    sys.exit()
    return nr.flatten(), ns.flatten()




def create_ast_from_xy(x, y, info):
    """
    Create AST records matching R/R+S = y

    :param x:
    :param y:
    :param info:
    :return:
    """



    # Get number of resistance records (ints)
    #nr = np.random.randint(1, 10, size=(1, x.shape[0]))
    #nr = np.ones(shape=(1, x.shape[0])).astype(int) * 500
    #ns = np.round((nr / y) - nr).astype(int)
    ns = np.ones(shape=(1, x.shape[0])).astype(int) + 2000
    nr = (ns / y).astype(int)
    #nr = ((y*ns) / (1-y)).astype(int)

    print('nr', nr)
    print('ns', ns)

    # Create model record
    v = list(info.values())
    r = [[i, 'resistant'] + v for i in x]
    s = [[i, 'sensitive'] + v for i in x]

    print(nr)
    print(ns)

    # Repeat n number of times
    r = np.repeat(r, nr.flatten(), axis=0)
    s = np.repeat(s, ns.flatten(), axis=0)

    # Return
    return np.vstack((r,s))


def straight_line(x, slope, intercept):
    ys = intercept + slope * x
    return x, ys


def a():
    pass


# Get dates between range
sdate = '2010-01-01'
edate = '2010-01-05'
d = pd.date_range(sdate, edate, freq='d')

# Compute sari function
x = np.arange(d.shape[0])
sari = x + 10

print("sari")
print(sari)

# Define sari values
#sari = np.array([0.09, 0.5, 0.57, 0.6, 0.57])

# Compute number of resistant and sensitive records
nr, ns = create_ast_from_values(sari, ns=sari)

# Create DataFrame
data = pd.DataFrame()
data['DATE'] = d
data['SPECIMEN'] = 's'
data['MICROORGANISM'] = 'o'
data['ANTIMICROBIAL'] = 'a'
data['ns'] = ns
data['nr'] = nr

# Create records by repeating entries and
# concatenating the results
r = data \
    .loc[data.index.repeat(data.nr)] \
    .assign(SENSITIVITY='resistant') \
    .reset_index(drop=True)

s = data \
    .loc[data.index.repeat(data.ns)] \
    .assign(SENSITIVITY='sensitive') \
    .reset_index(drop=True)


records = pd.concat([r, s], axis=0)


# ----------------------------
# Compute SARI
# ----------------------------
# Show
from pyamr.core.sari import SARI

# Configuration
shift, period = '1D', '1D'

# Compute the index
iti = SARI().compute(records, shift=shift,
    period=period, cdate='DATE')

# Display
iti = iti.reset_index()

# Show results
print("Data:")

plt.plot(sari)
plt.figure()
plt.plot(iti.index.values, iti.sari)
plt.show()

print(records)
import sys
sys.exit()












# Create
x = np.arange(d.shape[0])


#y1 = np.sin(2*x)
#mu, sigma = 0, 0.1
#y2 = x + np.random.normal(mu, sigma)
x1, y1 = x, np.sin(x) + 50
x3, y3 = straight_line(x, 0.5, 0.2)
y4 = np.array([10, 12, 24, 35, 48])

print(y3)
print(y4)

print(x3.shape, y3.shape)
print(d.shape, x.shape)



data = create_ast_from_xy(d, y1, CONFIGURATION[0])

# Create DataFrame
data = pd.DataFrame(data,
    columns=['DATE',
             'SENSITIVITY',
             'SPECIMEN',
             'MICROORGANISM',
             'ANTIMICROBIAL'])
#data['DATE'] = data.X.map(dict(zip(x,d.values)))

# Ensure is a datetime series.
data.DATE = pd.to_datetime(data.DATE)

# Show
print("\nData:")
print(data)
print("\nTypes")
print(data.dtypes)

# ----------------------------
# Compute SARI
# ----------------------------
# Show
from pyamr.core.sari import SARI

# Configuration
shift, period = '1D', '1D'

# Compute the index
iti = SARI().compute(data, shift=shift,
    period=period, cdate='DATE')

print(iti)

# Display
iti = iti.reset_index()

plt.plot(x, y1)
plt.figure()
plt.plot(iti.index.values, iti.sari)

plt.show()