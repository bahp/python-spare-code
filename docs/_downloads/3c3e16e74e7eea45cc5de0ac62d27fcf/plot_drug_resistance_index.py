"""
Drug Resistance Index
---------------------

The ``DRI`` measures changes through time in the proportion of disease-causing
pathogens that are resistant to the antibiotics commonly used to treat them.
The annual percentage change in the ``DRI`` is a measure of the rate of depletion
of antibiotic effectiveness.

Since antibiotic use may change over time in response to changing levels of
antibiotic resistance, we compare trends in the index with the counterfactual
case, where antibiotic use remains fixed to a baseline year. A static-use ``DRI``
allows assessment of the extent to which drug use has adapted in response to
resistance and the burden that this resistance would have caused if antibiotic
use patterns had not changed. Changing antibiotic use patterns over time may
mitigate the burden of antibiotic resistance. To incorporate changing trends
in antibiotic use, we also construct an adaptive version of the ``DRI``.

.. math::

    R\_fixed(i) = \sum_{k}{p_{i,k}^{t} * q_{i,k}^{0}}

.. math::

    R(i) = \sum_{k}{p_{i,k}^{t} * q_{i,k}^{t}}

where

  * **t** is time
  * **i** is the microorganism
  * **k** is the antimicrobial
  * **p** is the proportion of resistance among organism **i** to drug **k** at time t
  * **q** is the frequency of drug **k** used to treat organism in at time **t**.

See below for a few resources.

  * `R1`_: How to calculate DRI? (CDDEP)
  * `R2`_: New metric aims to simplify how global resistance is measured (POST)
  * `R3`_: Communicating trends in resistance using DRI.
  * `R4`_: Tracking global trends in the effectiveness of antibiotic therapy.
  * `R5`_: The proposed DRI is not a good measure of antibiotic effectiveness.

.. _R1: http://antimicrob.net/wp-content/uploads/DRI-instructions_engl.pdf
.. _R2: https://www.cidrap.umn.edu/antimicrobial-stewardship/new-metric-aims-simplify-how-global-resistance-measured
.. _R3: https://bmjopen.bmj.com/content/1/2/e000135
.. _R4: https://gh.bmj.com/content/4/2/e001315
.. _R5: https://gh.bmj.com/content/4/4/e001838


"""
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

def print_example_heading(n, t=''):
    print("\n" + "=" * 80 + "\nExample %s\n"%n + "=" * 80)

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False


######################################################################
# a) Example using CDDEP (tutorial)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The Drug Resistance Index or ``DRI``can be calculated at the facility or individual
# ward level to measure the performance of stewardship efforts. The index can
# be calculated in three simple steps using the pharmacy and antibiogram data
# available in most hospitals.
#
# The CDDEP recommends to aggregate susceptibility tests results by species
# and class of antimicrobials (see note). Laboratories and hospital antibiograms
# report susceptibility results by individual antibiotic compound. However, certain
# drugs may be widely tested to proxy for the effectiveness of an entire class in
# vitro, although they are rarely used in clinical practice (e.g. oxacillin tests
# for all b-lactam activity against staphylococci, nalidixic axic for quinolone
# activity in Gram-negative uropathogens). In addition, not all drugs on formulary
# may be included in laboratory testing. Therefore, when matching resistance data to
# antibiotic utilization, we recommend grouping results by broader therapeutic class.
# The optimal grouping will depend on many factors such as the species being included
# in the index, facility formulary and testing practices etc.

# This sample data was copied from the CDDEP reference (R1) to
# evaluate whether the Drug Resistance Index is being computed
# correctly.
cddep = [
    ['Q2 2011', 'E. Coli', 'AMINOPENICILLINS', 329, 139, 300, 1000],
    ['Q2 2011', 'E. Coli', 'B-LACTAM, INCREASED ACTIVITY', 554, 72,  250, 1000],
    ['Q2 2011', 'E. Coli', 'G3 CEPHALOSPORINS', 287, 3,  100, 1000],
    ['Q2 2011', 'E. Coli', 'QUINOLONES, SYSTEMIC', 293, 41,  250, 1000],
    ['Q2 2011', 'E. Coli', 'SULFA & TRIMETH COMB', 334, 75,  100, 1000],
    ['Q3 2011', 'E. Coli', 'AMINOPENICILLINS', 231, 101, 250, 1100],
    ['Q3 2011', 'E. Coli', 'B-LACTAM, INCREASED ACTIVITY', 408, 54, 300, 1100],
    ['Q3 2011', 'E. Coli', 'G3 CEPHALOSPORINS', 211, 3, 150, 1100],
    ['Q3 2011', 'E. Coli', 'QUINOLONES, SYSTEMIC', 218, 36, 300, 1100],
    ['Q3 2011', 'E. Coli', 'SULFA & TRIMETH COMB', 236, 55, 100, 1100]
]

# Create DataFrame
df1 = pd.DataFrame(cddep,
    columns=['time', 'org', 'abx', 'isolates', 'R', 'use', 'use_period'])

# Format DataFrame
df1['S'] = df1.isolates - df1.R
df1['r_rate'] = (df1.R / (df1.R + df1.S))    #.round(decimals=2)
df1['u_weight'] = (df1.use / df1.use_period) #.round(decimals=2)
df1['w_rate'] = (df1.r_rate * df1.u_weight)  #.round(decimals=3)
df1['dri'] = df1.groupby(by='time').w_rate.transform(lambda x: x.sum())

#%%
# Lets see the results
if TERMINAL:
    print_example_heading(n=1)
    print('\nResult (CDDEP):')
    print(df1)
df1.round(decimals=3)



######################################################################
# b) Example using CDDEP (raw)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the previous example we computed the Drug Resistance Index or ``DRI``
# using the summary table provided by CDDEP. While this table is useful
# to understand the implementation of the Drug Resistance Index, the
# values contained in the columns (e.g. isolates, R, use or use_period)
# need to be computed from the raw susceptibility test data. This example
# shows how to go from raw susceptibility test and prescription data to
# the summary table in order to compute the ``DRI``.

#%%
# First, let's create the raw data from the summary table provided by CDDEP.

def create_raw_data_from_summary_table(df):
    """Create raw susceptibility and prescription data from summary.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with the summary data

    Returns
    -------

    """
    # Variable
    keep = ['time', 'org', 'abx']

    # Indexes
    idxr = df.index.repeat(df.R)
    idxs = df.index.repeat(df.S)
    idxu = df.index.repeat(df.use)

    # Compute partial DataFrames
    r = df[keep].reindex(idxr).assign(sensitivity='resistant')
    s = df[keep].reindex(idxs).assign(sensitivity='sensitive')
    u = df[keep].reindex(idxu).assign(dose='standard')

    # Return
    return pd.concat([r, s]), u


# .. note:: Uncomment if you need to create a simulation of
#           the raw susceptibility and prescription data based
#           on the aggregated measurements.

# Create raw data.
susceptibility, prescription = \
    create_raw_data_from_summary_table(df1)

# Save
#susceptibility.to_csv('./data/cddep/susceptibility.csv')
#prescription.to_csv('./data/cddep/prescription.csv')

#%%
# Lets visualise microbiology data
if TERMINAL:
    print_example_heading(n=2)
    print('\nSusceptibility:')
    print(susceptibility)
susceptibility.head(5)

#%%
# Lets visualise the prescription data
if TERMINAL:
    print('\nPrescription:')
    print(prescription)
prescription.head(5)

#%%
# Lets compute the DRI

# -------------------------------------------
# Compute SARI
# -------------------------------------------
# Libraries
from pyamr.core.sari import SARI

# Create sari instance
sari = SARI(groupby=['time',
                     'org',
                     'abx',
                     'sensitivity'])

# Compute SARI
df2 = sari.compute(susceptibility, return_frequencies=True)

# -------------------------------------------
# Compute DRI
# -------------------------------------------

def compute_drug_resistance_index(summry, groupby,
                                  cu='use', cr='sari', ct='time'):
    """Computes the Drug Resistance Index

    Parameters
    ----------
    summry: pd.DataFrame
        The summary DataFrame with columns including use,
        resistance and time.
    groupby:
        The elements to groupby as described in pandas.
    cu: str
        Column name with use
    cr: str
        Column name with resistance
    ct: str
        Column name with time

    Returns
    -------
    """
    # Clone matrix
    m = summry.copy(deep=True)

    # Compute
    m['use_period'] = m \
        .groupby(level=groupby)[cu] \
        .transform(lambda x: x.sum())
    m['u_weight'] = (m[cu] / m.use_period)  # .round(decimals=2)
    m['w_rate'] = (m[cr] * m.u_weight)  # .round(decimals=3)
    m['dri'] = m \
        .groupby(by=ct).w_rate \
        .transform(lambda x: x.sum())

    # Return
    return m



# Compute the number of prescriptions
aux = prescription \
    .groupby(by=['time', 'org', 'abx']) \
    .dose.count().rename('use')

# Merge susceptibility and prescription data
df2 = df2.merge(aux, how='inner',
    left_on=['time', 'org', 'abx'],
    right_on=['time', 'org', 'abx'])

# Compute drug resistance index
df2 = compute_drug_resistance_index(df2, groupby=0)


#%%
# Lets see the results
if TERMINAL:
    print('\nResult (CDDEP):')
    print(df1)
df2.round(decimals=3)



######################################################################
# c) Example using MIMIC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this example, we will compute the Drug Resistance Index or ``DRI`` using
# the freely-available ``MIMIC`` database which provides both susceptibility
# test data and prescriptions for a large number of patients admitted from
# 2008 to 2019. Please note that for de-identification purpose the dates were
# shifted into the future.
#
#
# .. note:: In this example we will be counting each entry in the
#           prescription data as one 'standard' dose for clarity. However,
#           this could be implemented more accurately by computing
#           for example the complete dosage.

#%%
# First, lets load the susceptibility and prescription data

# subset
subset = [
    'time',
    'laboratory_number',
    'specimen_code',
    'microorganism_code',
    'antimicrobial_code',
    'sensitivity'
]

# -----------------------------
# Load susceptibility test data
# -----------------------------
# Load data
path = Path('./data/mimic')
filename = 'microbiologyevents.csv'

if (path / filename).exists():
    data = pd.read_csv(path / 'microbiologyevents.csv')

    # Rename columns
    data = data.rename(columns={
        'chartdate': 'time',
        'micro_specimen_id': 'laboratory_number',
        'spec_type_desc': 'specimen_code',
        'org_name': 'microorganism_code',
        'ab_name': 'antimicrobial_code',
        'interpretation': 'sensitivity'
    })

    # Format data
    data = data[subset]
    data = data.dropna(subset=subset, how='any')
    data.time = pd.to_datetime(data.time)
    data.sensitivity = data.sensitivity.replace({
        'S': 'sensitive',
        'R': 'resistant',
        'I': 'intermediate',
        'P': 'pass'
    })


    # ------------------
    # Load prescriptions
    # ------------------
    # Load prescription data (limited to first nrows).
    use = pd.read_csv(path / 'prescriptions.csv', nrows=100000)

    # Keep prescriptions which have also been tested
    aux = use.copy(deep=True)
    aux.drug = aux.drug.str.upper()
    aux.starttime = pd.to_datetime(aux.starttime)
    aux = aux[aux.drug.isin(data.antimicrobial_code.unique())]

    # Rename variables
    susceptibility, prescription = data, aux

    #%%
    # Lets visualise microbiology data
    if TERMINAL:
        print_example_heading(n=3)
        print('\nSusceptibility:')
        print(susceptibility)
    susceptibility.head(5)

    #%%
    # Lets visualise the prescription data
    if TERMINAL:
        print('\nPrescription:')
        print(prescription)
    prescription.head(5)


    # %%
    # Lets compute the DRI

    # -------------------------------------------
    # Compute SARI
    # -------------------------------------------

    # .. note:: These are some possible options to group
    #           data by a datetime column.
    #
    #             by year:    data.time.dt.year
    #             by quarter: data.time.dt.to_period('Q')
    #             by decade:  pd.Grouper(key='time', freq='10AS')
    #             by century: data.time.dt.year // 100


    # Libraries
    from pyamr.core.sari import SARI

    # Create sari instance
    sari = SARI(groupby=[data.time.dt.year,
                         #'specimen_code',
                         'microorganism_code',
                         'antimicrobial_code',
                         'sensitivity'])

    # Compute SARI
    df3 = sari.compute(susceptibility, return_frequencies=True)

    # .. note: Uncomment this line to create random use if the
    #          prescription data is not available and we want
    #          to simulate it.

    # Create random use
    #df3['use'] = np.random.randint(10, 600, df3.shape[0])

    # Compute the number of prescriptions
    aux = prescription \
        .groupby(by=[aux.starttime.dt.year, 'drug']) \
        .drug.count().rename('use')
    aux.index.names = ['time', 'antimicrobial_code']

    # Merge susceptibility and prescription data
    df3 = df3.reset_index().merge(aux.reset_index(),
        how='inner',
        left_on=['time', 'antimicrobial_code'],
        right_on=['time', 'antimicrobial_code'])

    # Format the summary DataFrame
    df3 = df3.set_index(['time', 'microorganism_code', 'antimicrobial_code'])

    # Compute drug resistance index
    df3 = compute_drug_resistance_index(df3, groupby=0)

    #%%
    # Lets see the results
    if TERMINAL:
        print('\nResult (MIMIC):')
        print(df1)
    df3[['freq',
         'sari',
         'use',
         'use_period',
         'u_weight',
         'w_rate',
         'dri']].round(decimals=3)



    #%%
    # Lets display DRI over time
    #

    # Libraries
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get first element by level.
    df4 = df3.groupby(level=[0]).first().reset_index()

    # Display using lineplot
    #sns.lineplot(data=df4.dri)

    # Display using plt.plot.
    #plt.plot(df4.index, df4.dri, linewidth=0.75, markersize=3, marker='o')

    # Display using relplot
    sns.relplot(data=df4.reset_index(), x='time', y='dri',
        #hue='event', style='event', col='region, palette='palette',
        height=4, aspect=3.0, kind='line',
        linewidth=0.75, markersize=3, marker='o'
    )

    # Show
    plt.show()

    #%%
    # .. note:: In ``MIMIC``, the deidentification process for structured data required
    #           the removal of dates. In particular, dates were shifted into the future
    #           by a random offset for each individual patient in a consistent manner to
    #           preserve intervals, resulting in stays which occur sometime between the
    #           years 2100 and 2200. Time of day, day of the week, and approximate
    #           seasonality were conserved during date shifting.

    ########################################################################
    # d) Extra code


    """
    # Define data
    data = [
    
        ['o1', 'a1', 2000, 0, 100, 10],
        ['o1', 'a2', 2000, 0, 100, 9],
        ['o1', 'a3', 2000, 0, 100, 8],
    
        ['o1', 'a1', 2001, 50, 50, 10],
        ['o1', 'a2', 2001, 60, 40, 9],
        ['o1', 'a3', 2001, 70, 30, 8],
    
        ['o1', 'a1', 2002, 50, 50, 10],
        ['o1', 'a2', 2002, 60, 40, 10],
        ['o1', 'a3', 2002, 70, 30, 10],
    
        ['o1', 'a1', 2003, 50, 50, 10],
        ['o1', 'a2', 2003, 60, 40, 11],
        ['o1', 'a3', 2003, 70, 30, 12],
    
        ['o1', 'a1', 2004, 50, 50, 16],
        ['o1', 'a2', 2004, 60, 40, 22],
        ['o1', 'a3', 2004, 70, 30, 30],
    
        ['o1', 'a1', 2005, 30, 70, 16],
        ['o1', 'a2', 2005, 30, 70, 22],
        ['o1', 'a3', 2005, 30, 70, 30],
    
        ['o1', 'a1', 2006, 50, 50, 16],
        ['o1', 'a2', 2006, 60, 40, 22],
        ['o1', 'a3', 2006, 70, 30, 22],
    
        ['o1', 'a1', 2007, 100, 0, 10],
        ['o1', 'a2', 2007, 100, 0, 9],
        ['o1', 'a3', 2007, 100, 0, 8],
    
        ['o2', 'a1', 2001, 50, 50, 16],
        ['o2', 'a2', 2001, 60, 40, 22],
        ['o2', 'a3', 2001, 70, 30, 22],
    ]
    
    
    # Create DataFrame
    df = pd.DataFrame(data,
        columns=['o', 'a', 'year', 'R', 'S', 'dose'])
    
    # Format DataFrame
    df.year = pd.to_datetime(df.year, format='%Y')
    df['r_prop'] = df.R / (df.R + df.S)                                         # resistance proportion
    df['d_prop'] = df.dose / df.groupby(by=['o', 'year']).dose.transform('sum') # dose proportion
    
    # Show
    print("Data:")
    print(df)
    
    # Example 1: Fixed
    # ----------------
    
    # Define p and q
    p = df.r_prop
    q = df.d_prop
    
    # Compute
    r0 = (p * q).sum()
    r1 = np.dot(p, q)
    r2 = np.matmul(p, q)
    
    # Show
    print("\n" + "="*80 + "\nExample 1\n" + "="*80)
    print(r0)
    print(r1)
    print(r2)
    
    
    
    # Example 2: Manual
    # -----------------
    
    # Initialize
    qik0 = None
    v_fixed = []
    v_evolv = []
    
    # Loop
    for i,g in df.groupby(by=['o', 'year']):
        if qik0 is None:
            qik0 = list(g.groupby(by='year'))[0][1].d_prop
    
        v_fixed.append({'o': i[0], 't': i[1], 'v': np.dot(g.r_prop, qik0.T)})
        v_evolv.append({'o': i[0], 't': i[1], 'v': np.dot(g.r_prop, g.d_prop)})
    
    # Format
    v_fixed = pd.DataFrame(v_fixed)
    v_evolv = pd.DataFrame(v_evolv)
    v_combn = v_evolv.merge(v_fixed, how='outer',
        on=['o', 't'], suffixes=('_evolv', '_fixed'))
    
    # Show
    print("\n" + "="*80 + "\nExample 2\n" + "="*80)
    print(v_combn)
    
    
    
    
    # Example 3: Matrix
    # -----------------
    #
    
    print("\n" + "="*80 + "\nExample 3\n" + "="*80)
    
    
    m1 = pd.pivot_table(df, index='o', columns='a', values='r_prop')
    m2 = pd.pivot_table(df, index='o', columns='a', values='d_prop')
    print(m1)
    print(m2)
    print(m1*m2)
    
    # Loop
    for i,g in df.groupby(by='year'):
        m1 = pd.pivot_table(g, index='o', columns='a', values='r_prop')
        m2 = pd.pivot_table(g, index='o', columns='a', values='d_prop')
        dri = np.dot(m1, m2.T)
        print(i, dri)
        
        
    """

    """
    # Compute
    df3['use_period'] = df3 \
        .groupby(level=0).use \
        .transform(lambda x: x.sum())
    df3['u_weight'] = (df3.use / df3.use_period) #.round(decimals=2)
    df3['w_rate'] = (df3.sari * df3.u_weight)  #.round(decimals=3)
    df3['dri'] = df3 \
        .groupby(by='time').w_rate \
        .transform(lambda x: x.sum())
    """