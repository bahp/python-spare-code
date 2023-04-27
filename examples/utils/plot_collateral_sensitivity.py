"""
Mutual Information Criteria
---------------------------
"""
# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

#
from pathlib import Path
from itertools import combinations
from mic import mutual_info_matrix_v3

# See https://matplotlib.org/devdocs/users/explain/customizing.html
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

def collateral_resistance_index(m):
    """Collateral Resistance Index

    The collateral resistance index is based on the mutual
    information matrix. This implementation assumes there
    are two classes resistant (R) and sensitive (S).

    Parameters
    ----------
    m: np.array
        A numpy array with the mutual information matrix.

    Returns
    -------
    """
    return (m[0, 0] + m[1, 1]) - (m[0, 1] + m[1, 0])

def CRI(x, func):
    ct = np.array([[x.SS, x.SR], [x.RS, x.RR]])
    m = func(ct=ct)
    return collateral_resistance_index(m)



def combo_v1():
    # Build combination
    c = pd.DataFrame()
    for i, g in df.groupby(['o', 's']):
        for index in list(combinations(g.index, 2)):
            i, j = index
            s = pd.Series({
                'o': df.loc[i, 'o'],
                's': df.loc[i, 's'],
                'ax': df.loc[i, 'a'],
                'ay': df.loc[j, 'a'],
                'rx': df.loc[i, 'r'],
                'ry': df.loc[j, 'r']
            })
            c = pd.concat([c, s.to_frame().T])
            # c.append(s)

    """
    # Build combination
    c = pd.DataFrame()
    for i, g in data.groupby(['specimen_code',
                              'microorganism_code',
                              'laboratory_number']):
        for index in list(combinations(g.index, 2)):
            i, j = index
            s = pd.Series({
                'o': data.loc[i, 'microorganism_code'],
                's': data.loc[i, 'laboratory_number'],
                'ax': data.loc[i, 'antimicrobial_code'],
                'ay': data.loc[j, 'antimicrobial_code'],
                'rx': data.loc[i, 'sensitivity'],
                'ry': data.loc[j, 'sensitivity']
            })
            c = pd.concat([c, s.to_frame().T])


    # Add class
    c['class'] = c.rx + c.ry
    """
    return c


def create_df_combo_v1(d, col_o='o',
                          col_s='s',
                          col_a='a',
                          col_r='r'):
    """"""
    # Build combination
    c = []
    for i, g in d.groupby([col_o, col_s]):

        for i, j in combinations(g.sort_values(by=col_a).index, 2):
            s = pd.Series({
                'o': g.loc[i, col_o],
                's': g.loc[i, col_s],
                'ax': g.loc[i, col_a],
                'ay': g.loc[j, col_a],
                'rx': g.loc[i, col_r],
                'ry': g.loc[j, col_r]
            })
            c.append(s)

    # Concatenate
    c = pd.concat(c, axis=1).T

    # Add class
    c['class'] = c.rx + c.ry

    # Return
    return c

##################################################################
# a) A simple example
#

# Create matrix
data = [
    ['s1', 'o1', 'a1', 'S'],
    ['s1', 'o1', 'a2', 'S'],
    ['s1', 'o1', 'a3', 'R'],
    ['s2', 'o1', 'a1', 'S'],
    ['s2', 'o1', 'a2', 'S'],
    ['s2', 'o1', 'a3', 'R'],
    ['s2', 'o1', 'a4', 'R'],
    ['s3', 'o1', 'a1', 'R'],
    ['s3', 'o1', 'a2', 'S'],
    ['s4', 'o1', 'a2', 'R'],
    ['s4', 'o1', 'a1', 'S']
]

# Create DataFrame
df = pd.DataFrame(data,
    columns=['s', 'o', 'a', 'r'])

# Show
print("\nData:")
print(df)

# Create combo
c = create_df_combo_v1(df)

# Show
print("\nCombinations (within isolates):")
print(c)

# Build contingency
r = c.groupby(['ax', 'ay', 'class']).size().unstack()

# Show
print("\nContingency:")
print(r)

# Compute CRI
r['MIS'] = r.apply(CRI, args=(mutual_info_matrix_v3,), axis=1)

# Show
print("\n" + "="*80 + "\nExample 1\n" + "="*80)
print("\nResult")
print(r)



##################################################################
# b) Using NHS data
# ~~~~~~~~~~~~~~~~~
#
# .. note:: This method is too slow!

def load_sample(**kwargs):
    """"""
    path = '../../datasets/susceptibility-nhs/'
    path += 'susceptibility-v0.0.1/susceptibility-2009.csv'
    data = pd.read_csv(path, **kwargs)
    return data

def load_multiple(**kwargs):
    """"""
    path = '../../datasets/susceptibility-nhs/susceptibility-v0.0.1'
    files = Path(path).glob('susceptibility-*.csv')
    data = pd.concat([
        pd.read_csv(f, **kwargs)
            for f in files])
    return data

# Load data
data = load_sample(nrows=100000)
#data = load_multiple(nrows=1000)

# Format
data.sensitivity = data.sensitivity\
    .replace({
        'sensitive': 'S',
        'resistant': 'R'
    })

# Select specimen
data = data[data.specimen_code.isin(['BLDCUL'])]
data = data[data.microorganism_code.isin(['SAUR'])]

# Create combo
c = create_df_combo_v1(data,
    col_o='microorganism_code',
    col_s='laboratory_number',
    col_a='antimicrobial_code',
    col_r='sensitivity')

# Build contingency
r = c.groupby(['ax', 'ay', 'class']).size().unstack()

# Compute CRI
r['MIS'] = r.fillna(0) \
    .apply(CRI, args=(mutual_info_matrix_v3,), axis=1)

# Show
print("\n" + "="*80 + "\nExample 2\n" + "="*80)
print("\nResult")
print(r)

# Reformat
aux = r.reset_index() \
    .pivot(columns='ax',
           index='ay',
           values='MIS')

# Display
sns.heatmap(data=aux*100, annot=True, linewidth=.5,
    cmap='coolwarm', vmin=-70, vmax=70, center=0,
    square=True)

# Show
plt.tight_layout()
plt.show()