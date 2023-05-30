"""
Collateral Sensitivity Index
-----------------------------

In order to run the script in such a way that we can profile
the time used for each method, statement, .. use the following
command:

    $ python -m cProfile -s cumtime plot_collateral_sensitivity.py > outcome.csv
"""
# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

#
from pathlib import Path
from datetime import datetime
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


def create_df_combo_v1(d, col_o='o', # organism
                          col_s='s', # sample
                          col_a='a', # antimicrobial
                          col_r='r'): # outcome / result
    """

    .. note:: There might be an issue if there are two different outcomes
              for the same record. For example, a susceptibility test
              record for penicillin (APEN) with R and another one with
              S. Warn of this issue if it appears!

    :param d:
    :param col_o:
    :param col_s:
    :param col_a:
    :param col_r:
    :return:
    """

    # This is innefficient!

    # Build combination
    c = []
    for i, g in d.groupby([col_s, col_o]):
        for x, y in combinations(g.sort_values(by=col_a).index, 2):
            s = pd.Series({
                'o': g.loc[x, col_o],
                's': g.loc[x, col_s],
                'ax': g.loc[x, col_a],
                'ay': g.loc[y, col_a],
                'rx': g.loc[x, col_r],
                'ry': g.loc[y, col_r]
            })
            c.append(s)


    # Concatenate
    c = pd.concat(c, axis=1).T

    # Add class
    c['class'] = c.rx + c.ry

    # Return
    return c


def create_combinations_v1(d, col_specimen='s',
                              col_lab_id='l',
                              col_microorganism='o',
                              col_antimicrobial='a',
                              col_result='r'):
    """Creates the dataframe with all combinations.

    Parameters
    ----------

    Returns
    --------
    """
    # Initialize
    c = []

    # Loop
    for i, g in d.groupby([col_specimen,
                           col_microorganism,
                           col_lab_id]):
        for x, y in combinations(g.sort_values(by=col_antimicrobial).index, 2):
            c.append({
                'specimen': g.loc[x, col_specimen],
                'lab_id': g.loc[x, col_lab_id],
                'o': g.loc[x, col_microorganism],
                'ax': g.loc[x, col_antimicrobial],
                'ay': g.loc[y, col_antimicrobial],
                'rx': g.loc[x, col_result],
                'ry': g.loc[y, col_result]
            })

    # Create DataFrame
    c = pd.DataFrame(c)
    # Add class
    c['class'] = c.rx + c.ry
    # Return
    return c


def create_combinations_v2(d, col_o='o',
      col_s='s', col_a='a', col_r='r'):
    """Creates the dataframe with all combinations.

    .. note:: There might be an issue if there are two different outcomes
              for the same record. For example, a susceptibility test
              record for penicillin (APEN) with R and another one with
              S. Warn of this issue if it appears!

    Parameters
    ----------

    Returns
    --------

    """
    # Initialize
    c = pd.DataFrame()

    # Loop
    for i, g in d.groupby([col_s, col_o]):

        aux = []
        for x, y in combinations(g.sort_values(by=col_a).index, 2):
            aux.append({
                'ax': g.loc[x, col_a],
                'ay': g.loc[y, col_a],
                'rx': g.loc[x, col_r],
                'ry': g.loc[y, col_r]
            })
        aux = pd.DataFrame(aux)
        aux['s'] = i[0]
        aux['o'] = i[1]

        # Concatenate
        c = pd.concat([c, aux], axis=0)

    # Add class
    c['class'] = c.rx + c.ry

    # Return
    return c

##################################################################
# a) A basic example
#
# Note that the columns names are the initial for the following
# full names: s=specimen, l=laboratory sample, o=organism,
# a=antimicrobial and r=result

# Create matrix
data = [
    ['s1', 'l1', 'o1', 'a1', 'S'],
    ['s1', 'l1', 'o1', 'a2', 'S'],
    ['s1', 'l1', 'o1', 'a3', 'R'],
    ['s1', 'l2', 'o1', 'a1', 'S'],
    ['s1', 'l2', 'o1', 'a2', 'S'],
    ['s1', 'l2', 'o1', 'a3', 'R'],
    ['s1', 'l2', 'o1', 'a4', 'R'],
    ['s1', 'l3', 'o1', 'a1', 'R'],
    ['s1', 'l3', 'o1', 'a2', 'S'],
    ['s1', 'l4', 'o1', 'a2', 'R'],
    ['s1', 'l4', 'o1', 'a1', 'S'],
    ['s1', 'l5', 'o1', 'a5', 'S'],
    ['s1', 'l6', 'o1', 'a4', 'S'],
    ['s1', 'l5', 'o1', 'a2', 'S'],
]

# Create DataFrame
df = pd.DataFrame(data,
    columns=['s', 'l', 'o', 'a', 'r'])

# Show
print("\nData:")
print(df)

# Create combo
c = create_combinations_v1(df)

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

# Create index with all pairs
index = pd.MultiIndex.from_product(
    [df.a.unique(), df.a.unique()]
)

# Reformat
aux = r['MIS'] \
    .reindex(index, fill_value=np.nan)\
    .unstack()

# Display
sns.heatmap(data=aux*100, annot=True, linewidth=.5,
    cmap='coolwarm', vmin=-70, vmax=70, center=0,
    square=True)

# Show
plt.tight_layout()
#plt.show()



##################################################################
# b) Using NHS data
# ~~~~~~~~~~~~~~~~~
#


def load_susceptibility_nhs(**kwargs):
    """Load and format MIMIC microbiology data.

    Parameters
    ----------
    **kwargs: dict-like
        The arguments as used in pandas read_csv function

    Returns
    --------
    """
    # Load data
    path = Path('../../datasets/susceptibility-nhs/')
    path = path / 'susceptibility-v0.0.1'

    data = pd.concat([
        pd.read_csv(f, **kwargs)
            for f in Path(path).glob('susceptibility-*.csv')])

    # Format data
    data.sensitivity = data.sensitivity \
        .replace({
            'sensitive': 'S',
            'resistant': 'R',
            'intermediate': 'I',
            'highly resistant': 'HR'
    })

    # Select specimen
    # data = data[data.specimen_code.isin(['URICUL'])]
    # data = data[data.microorganism_code.isin(['SAUR', 'ECOL', 'PAER'])]
    # data = data[data.sensitivity.isin(['R', 'S'])]
    # data = data[data.laboratory_number.isin(['H1954180', 'M1596362'])]

    data = data[data.sensitivity.isin(['R', 'S', 'I', 'HR'])]

    # .. note:: For some reason, for the same specimen and antimicrobial
    #           there are sometimes contradictory outcomes (e.g. R and S)
    #           so we are removing this by keeping the last.

    # Keep only last/first specimen (sometimes repeated)
    subset = data.columns.tolist()
    subset = subset.remove('sensitivity')
    data = data.drop_duplicates(subset=subset, keep='last')

    # Further cleaning

    # Return
    return data


def load_susceptibility_mimic(**kwargs):
    """Load and format MIMIC microbiology data.

    Parameters
    ----------
    **kwargs: dict-like
        The arguments as used in pandas read_csv function

    Returns
    --------
    """
    # Load data
    path = Path('../../datasets/susceptibility-mimic/')
    path = path / 'microbiologyevents.csv'
    data = pd.read_csv(path, **kwargs)

    # Format data
    data = data.rename(columns={
        'micro_specimen_id': 'laboratory_number',
        'spec_type_desc': 'specimen_code',
        'org_name': 'microorganism_code',
        'ab_name': 'antimicrobial_code',
        'interpretation': 'sensitivity'
    })

    # Keep only last/first specimen

    # Remove inconsistent records, for example if for an specimen there are two
    # rows for the same antimicrobial. Or even worse, these two rows are
    # contradictory (e.g. R and S)

    # Other cleaning.

    # Return
    return data


"""
# Load data
#data = load_susceptibility_mimic()
data = load_susceptibility_nhs()

# Create combo
c = create_combinations_v1(data,
    col_specimen='specimen_code',
    col_lab_id='laboratory_number',
    col_microorganism='microorganism_code',
    col_antimicrobial='antimicrobial_code',
    col_result='sensitivity')

# Create folder if it does not exist.
today = datetime.now().strftime("%Y%m%d-%H%M%S")
path = Path('./outputs/cri/') / today
Path(path).mkdir(parents=True, exist_ok=True)

# Save combinations file.
c.to_csv(path / 'combinations.csv')

# Build contingency
r = c.groupby(['specimen', 'o', 'ax', 'ay', 'class']).size().unstack()

# Compute CRI
r['MIS'] = r.fillna(0) \
    .apply(CRI, args=(mutual_info_matrix_v3,), axis=1)

# Show
print("\n" + "="*80 + "\nExample 2\n" + "="*80)
print("\nResult")
print(r)

# Save collateral sensitivity index file.
r.to_csv(path / 'contingency.csv')
"""