"""
Generate data AST
-----------------

"""

# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
CONFIGURATION = [
    {
        'date': ['2010-01-01', '2018-01-05'],
        'info': {
            'SPECIMEN': 'BLDCUL',
            'MICROORGANISM': 'SAUR',
            'ANTIMICROBIAL': 'ATAZ',
        },
        'kwargs': {
            'line_offset': 80,
            'line_slope': -0.3,
            'sin_amplitude': 5
        }
    },
    {
        'date': ['2010-01-01', '2018-01-05'],
        'info': {
            'SPECIMEN': 'BLDCUL',
            'MICROORGANISM': 'SAUR',
            'ANTIMICROBIAL': 'ACIP',
        },
    },
    {
        'date': ['2010-05-01', '2019-01-05'],
        'info': {
            'SPECIMEN': 'BLDCUL',
            'MICROORGANISM': 'ECOL',
            'ANTIMICROBIAL': 'AMER',
        },
        'kwargs': {
            'line_offset': 80,
            'line_slope': -0.01,
            'sin_amplitude': 1,
            'noise_amplitude': 1
        }
    },
    {
        'date': ['2010-05-01', '2019-01-05'],
        'info': {
            'SPECIMEN': 'URICUL',
            'MICROORGANISM': 'ECOL',
            'ANTIMICROBIAL': 'AMER',
        },
        'kwargs': {
            'line_offset': 10,
            'line_slope': 0.2,
            'sin_amplitude': 3
        }
    },
    {
        'date': ['2010-05-01', '2018-01-05'],
        'info': {
            'SPECIMEN': 'URICUL',
            'MICROORGANISM': 'ECOL',
            'ANTIMICROBIAL': 'ACIP',
        },
        'kwargs': {
            'sin_amplitude': 3,
            'sin_stretch': 3
        }
    },
    {
        'date': ['2010-05-01', '2020-01-05'],
        'info': {
            'SPECIMEN': 'URICUL',
            'MICROORGANISM': 'ECOL',
            'ANTIMICROBIAL': 'ATAZ',
        },
        'kwargs': {
            'line_offset': 80,
            'line_slope': 0.3,
            'sin_amplitude': 0,
            'jump_start': 30,
            'jump_amplitude': 5
        }
    },
    {
        'date': ['2012-05-01', '2018-01-05'],
        'info': {
            'SPECIMEN': 'URICUL',
            'MICROORGANISM': 'SAUR',
            'ANTIMICROBIAL': 'AMER',
        },
        'kwargs': {
            'line_offset': 20,
            'line_slope': 0.1,
            'sin_amplitude': 0,
            'noise_amplitude': 0
        }
    },
]

def make_series(n_samples,
                line_slope=0.2,
                line_offset=30,
                sin_amplitude=1,
                sin_stretch=1,
                sin_c=1,
                noise_amplitude=2,
                random=False,
                jump_start=None,
                jump_amplitude=5):
    """Create series...

    f(x) = a * sin( b (x+c)) + d

    """
    # Configuration
    if random:
        line_slope = np.random.randint(10, 20) / 100
        line_offset = np.random.randint(40, 60)
        sin_amplitude = np.random.randint(1, 60) / 100
        sin_stretch = np.random.randint(3, 10)
        sin_c = np.random.randint(5, 10)
        noise_amplitude = np.random.randint(10, 20)
        jump_start = np.random.randint(0, 2)
        jump_amplitude = np.random.randint(5, 10)

    # Create components and final function
    x = np.linspace(0, 50, n_samples)
    line = (line_slope * x) + line_offset
    noise = noise_amplitude * np.random.rand(len(x))
    season = sin_amplitude * np.sin(sin_stretch * x)
    y = line + season + noise + x/5

    # Add a jump.
    if jump_start is not None:
        y[jump_start:] += jump_amplitude

    # .. note: This is done to avoid having extreme
    #          values which end up giving a negative
    #          number of R or S. In this case it will
    #          have the same shape but will be using all
    #          the y-axis from 0.2 to 0.8

    # Normalize within the range (20, 80)
    rmin, rmax = 20, 80
    if (max(y) > rmax) or (min(y) < rmin):
        y = ((y - min(y)) / (max(y) - min(y)))
        y = y * (rmax - rmin) + rmin

    # Return
    return y


def create_ast_from_values(cfg, nr=30): #y, dates, info, nr=None):
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
    # Generate dates
    sdate = cfg.get('date')[0]
    edate = cfg.get('date')[1]
    dates = pd.date_range(sdate, edate, freq='m')

    # Generate y function
    kwargs = cfg.get('kwargs', {})
    random = not bool(kwargs)
    y = make_series(dates.shape[0], random=random, **kwargs)

    if nr is None:
       nr = np.random.randint(10, 100, size=y.shape[0])

    # Compute number of S based on number of R.
    ns = (((100/y) - 1) * nr).astype(int)

    # Create DataFrame
    data = pd.DataFrame(data=dates, columns=['DATE'])

    # Fill rest of information
    for k, v in cfg.get('info').items():
        data[k] = v
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

    # Return
    return records


# Constants
SPECIMENS = ['BLDCUL', 'URICUL']
MICROORGANISMS = ['ECOL', 'KPNE']
ANTIMICROBIALS = ['ATAZ', 'AMER', 'ACIP']


# Compute records
records = pd.concat([
    create_ast_from_values(cfg)
        for cfg in CONFIGURATION])


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
print(iti)

# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(iti, row="ANTIMICROBIAL",
    hue="SPECIMEN", col='MICROORGANISM',
    palette="tab20c", height=2, aspect=3)

# Draw a horizontal line to show the starting point
grid.refline(y=0.5, linewidth=0.75)

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "DATE", "sari",
    linewidth=0.75, marker="o", markersize=1)

# Show
plt.tight_layout()
plt.show()