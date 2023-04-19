"""
Main 06 - Plot Treemap with MIMIC
---------------------------------

This example displays a Treemap using the MIMIC dataset.

.. warning:: It is not completed!
"""

# Libraries
import pandas as pd


from plotly.io import show

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False


# ---------------------
# Helper method
# ---------------------

# Methods
def build_hierarchical_dataframe(df, levels, value_column, color_columns=None):
    """
    Build a hierarchy of levels for Sunburst or Treemap charts.

    Levels are given starting from the bottom to the top of the hierarchy,
    ie the last level corresponds to the root.
    """
    df_all_trees = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])
    for i, level in enumerate(levels):
        df_tree = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])
        dfg = df.groupby(levels[i:]).sum()
        dfg = dfg.reset_index()
        df_tree['id'] = dfg[level].copy()
        if i < len(levels) - 1:
            df_tree['parent'] = dfg[levels[i+1]].copy()
        else:
            df_tree['parent'] = 'total'
        df_tree['value'] = dfg[value_column]
        #df_tree['color'] = dfg[color_columns[0]] / dfg[color_columns[1]]
        df_all_trees = df_all_trees.append(df_tree, ignore_index=True)
    #total = pd.Series(dict(id='total', parent='',
    #                          value=df[value_column].sum(),
    #                          color=df[color_columns[0]].sum() / df[color_columns[1]].sum()))
    total = pd.Series(dict(id='total', parent='', value=df[value_column].sum()))
    df_all_trees = df_all_trees.append(total, ignore_index=True)
    return df_all_trees

def load_sunburst():
    """Load sunburst data."""
    # Define URL
    URL = 'https://raw.githubusercontent.com/plotly/'
    URL+= 'datasets/96c0bd/sunburst-coffee-flavors-complete.csv'
    # Load dataframe
    df = pd.read_csv(URL)
    # Return
    return df

def load_microbiology_nhs(n=10000):
    """Loads and formats microbiology data."""
    # Libraries
    from pyamr.core.sari import sari
    from pyamr.datasets.load import load_data_nhs

    # Load data
    data, antimicrobials, microorganisms = \
        load_data_nhs(nrows=n)

    data = data[data.specimen_code.isin(['BLOOD CULTURE'])]

    # Create DataFrame
    dataframe = data.groupby(['specimen_code',
                              'microorganism_code',
                              'antimicrobial_code',
                              'sensitivity']) \
        .size().unstack().fillna(0)


    # Compute frequency
    dataframe['freq'] = dataframe.sum(axis=1)
    dataframe['sari'] = sari(dataframe, strategy='hard')
    dataframe['sari_medium'] = sari(dataframe, strategy='medium')
    dataframe['sari_soft'] = sari(dataframe, strategy='soft')
    dataframe = dataframe.reset_index()

    # Add info for popup (micro and abxs)
    dataframe = dataframe.merge(antimicrobials,
        how='left', left_on='antimicrobial_code',
        right_on='antimicrobial_code')
    dataframe = dataframe.merge(microorganisms,
        how='left', left_on='microorganism_code',
        right_on='microorganism_code')

    # Format dataframe
    dataframe = dataframe.round(decimals=3)

    # Configuration
    LEVELS = ['specimen_code', 'microorganism_code', 'antimicrobial_code']
    COLORS = ['sari']
    VALUE = 'freq'

    dataframe = dataframe[LEVELS + COLORS + [VALUE]]

    aux2 = dataframe.groupby(LEVELS).agg('sum').reset_index()


    # Return
    aux = build_hierarchical_dataframe(aux2, LEVELS, COLORS, VALUE)

    return aux


# -----------------------------------
# Display basic
# -----------------------------------
# Libraries
import plotly.graph_objects as go

# Load data
df = load_sunburst()

# Show data
print("\nData:")
print(df)

# Define template
htmp = '<b>%{label}</b><br>'
htmp+= 'Sales:%{value}<br>'
htmp+= 'Success rate: %{color:.2f}'

# Create figure
fig = go.Figure(go.Treemap(
    ids=df.ids,
    labels=df.labels,
    parents=df.parents,
    pathbar_textfont_size=15,
    root_color="lightgrey",
    #maxdepth=3,
    branchvalues='total',
    #marker=dict(
    #    colors=df_all_trees['color'],
    #    colorscale='RdBu',
    #    cmid=average_score),
    #hovertemplate=htmp,
    #marker_colorscale='Blues'
))

# Update layout
fig.update_layout(
    uniformtext=dict(minsize=10, mode='hide'),
    margin = dict(t=50, l=25, r=25, b=25)
)


# -----------------------------------
# Display NHS
# -----------------------------------
# Load data
#df = load_microbiology_nhs()
#df = df.drop_duplicates(subset=['id', 'parent'])

# Show
print("DF:")
print(df)

# Create data
df = pd.DataFrame()
df['id'] = ['BLD', 'SAUR', 'PENI', 'CIPRO']
df['parent'] = [None, 'BLD', 'SAUR', 'SAUR']
df['value'] = [0, 1, 2, 3]
df['info'] = ['info1', 'info2', 'info3', 'info4']

# Define template
htmp = '<b> %{id} </b><br>'
htmp+= 'Info: %{info} <br>'
htmp+= 'Value %{value}'

# Create figure
fig = go.Figure(go.Treemap(
    #ids=df.id,
    labels=df.id,
    values=df.value,
    parents=df.parent,
    pathbar_textfont_size=15,
    root_color="lightgrey",
    #maxdepth=3,
    branchvalues='total',
    #marker=dict(
    #    colors=df_all_trees['color'],
    #    colorscale='RdBu',
    #    cmid=average_score),
    #hovertemplate=htmp,
    #marker_colorscale='Blues'
    #hovertemplate=htmp, # overrides hoverinfo
    #texttemplate=htmp, # overrides textinfo
    #hoverinfo=['all'],
    #textinfo=['all']
    textinfo="label+value",
))

# Update layout
fig.update_layout(
    uniformtext=dict(minsize=10, mode='hide'),
    margin = dict(t=50, l=25, r=25, b=25)
)

"""
# Add traces.
fig.add_trace(go.Treemap(
    #labels=df_trees.id,
    #parents=df_trees.parent,
    #values=df_trees.value,
    #branchvalues='total',
    labels=df_trees.ids,
    parents=df_trees.parents,
    values=df_trees.values,
    #marker=dict(
    #    colors=df_all_trees['color'],
    #    colorscale='RdBu',
    #    cmid=average_score),
    #hovertemplate='<b>%{label} </b> <br> Sales: %{value}<br> Success rate: %{color:.2f}',
    #name=''
    ), 1, 1)

# Update layout
#fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
"""

# Show
show(fig)