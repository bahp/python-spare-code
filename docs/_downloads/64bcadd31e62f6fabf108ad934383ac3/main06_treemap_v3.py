"""
06. Plot Treemap with NHS
-------------------------------

This example displays a Treemap using a portion of the NHS dataset. This
example needs ``pyAMR`` to load the corresponding data.

.. warning:: It is not completed!
.. warning:: It might take some time to load.
.. note:: It uses ``plotly.express`` rather than ``go.Treemap``.
.. note:: https://plotly.com/python/treemaps/

"""
# Plotly

import numpy as np
import pandas as pd
import plotly.express as px

from plotly.io import show

# Import own libraries
from pyamr.core.sari import sari
from pyamr.datasets.load import load_data_nhs

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False


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
        df_tree['color'] = dfg[color_columns[0]]
        df_all_trees = df_all_trees.append(df_tree, ignore_index=True)
    total = pd.Series(dict(id='total', parent='',
        value=df[value_column].sum(),
        color=df[color_columns[0]].sum()))
    df_all_trees = df_all_trees.append(total, ignore_index=True)
    return df_all_trees


TERMINAL = True

"""
# --------------------------------------------------------------------
#                               Main
# --------------------------------------------------------------------
# Load data
data, antimicrobials, microorganisms = load_data_nhs(nrows=10000)

# Create DataFrame
dataframe = data.groupby(['specimen_code',
                          'microorganism_code',
                          'antimicrobial_code',
                          'sensitivity']) \
                .size().unstack().fillna(0)

# Compute frequency
dataframe['freq'] = dataframe.sum(axis=1)

# Compute sari
dataframe['sari'] = sari(dataframe, strategy='hard')
dataframe['sari_medium'] = sari(dataframe, strategy='medium')
dataframe['sari_soft'] = sari(dataframe, strategy='soft')

# Reset index
dataframe = dataframe.reset_index()

# --------------------------------------------
# Add info for popup
# --------------------------------------------
dataframe = dataframe.merge(antimicrobials,
    how='left', left_on='antimicrobial_code',
    right_on='antimicrobial_code')

# Add antimicrobials information
dataframe = dataframe.merge(microorganisms,
    how='left', left_on='microorganism_code',
    right_on='microorganism_code')

# Format dataframe
dataframe = dataframe.round(decimals=3)

# Replace
dataframe.microorganism_name = \
    dataframe.microorganism_name.str.title()
dataframe.columns = \
    dataframe.columns.str.title().str.replace('_', ' ')


# Show
if TERMINAL:
    print("\nColumns:")
    print(dataframe.dtypes)
    print("\nDF:")
    print(dataframe)


# -------------------------------------------
# Plot
# -------------------------------------------
def guide_template(names):
    return '<br>'.join([
        "%2d <b>%-45s</b> %%{customdata[%s]}" % (i, n, i )
            for i, n in enumerate(names)])

# Guide template
htmp_guide = guide_template(dataframe.columns.tolist())

# Define own template
htmp = "Specimen:      (%{customdata[0]})<br>"
htmp+= "Microorganism: %{customdata[26]} (%{customdata[1]})<br>"
htmp+= "Antimicrobial: %{customdata[12]} (%{customdata[2]})<br>"
htmp+= "Freq: %{customdata[8]}<br>"
htmp+= "SARI: %{customdata[9]}<br>"

htmp = "(%{customdata[0]})<br>"
htmp+= "%{customdata[26]} (%{customdata[1]})<br>"
htmp+= "%{customdata[12]} (%{customdata[2]})<br>"
htmp+= "Freq: %{customdata[8]}<br>"
htmp+= "SARI: %{customdata[9]}<br>"

# Display
fig = px.treemap(dataframe,
    path=['Specimen Code',
          'Microorganism Code',
          'Antimicrobial Code'],
    #hover_name=,
    hover_data=dataframe.columns.tolist(),
    values='Freq',
    color='Sari',
    color_continuous_scale='Reds',
    title='Treemap of <Microorganisms, Antimicrobials> pairs')


# Show current template
print(fig.data[0].hovertemplate)
"""

"""
The default hover template looks as follows:

    labels=%{label}<br>
    Freq=%{value}<br>
    parent=%{parent}<br>
    id=%{id}<br>
    Microorganism Name=%{customdata[0]}<br>
    Name=%{customdata[1]}<br>
    Sari Medium=%{customdata[2]}<br>
    Sari Soft=%{customdata[3]}<br>
    Sari=%{color}
    <extra></extra>
"""

"""
# Set custom data (not needed and inconsistent)
#fig.data[0].customdata = dataframe.to_numpy()
#fig.data[0].hovertemplate = htmp

# Uncomment to check the customdata[i] information
#fig.data[0].hovertemplate = \
#    guide_template(dataframe.columns.tolist())

# Update: I
#fig.update_traces(hovertemplate='labels=%{label}')
#fig.update_traces(texttemplate='Freq=%{value:.2f}<br>')
fig.update_traces(hovertemplate=htmp_guide)
#fig.update_traces(hovertemplate=htmp)

# Update: II
# But it seems to me you want something like
#fig.data[0].hovertemplate = '%{label}<br>%{value}'
#fig.data[0].hovertemplate = '%{Freq}<br>%{Antimicrobial Name}'

fig.update_layout(
    margin={
        'l': 0,
        'r': 0,
        'b': 0,
        't': 0,
        'pad': 4
     })
# ----------------------------
# Save
# ----------------------------
# Libraries
import time
from pathlib import Path

# Define pipeline path
path = Path('./objects') / 'plot_main08_treemap'
filename = '%s.html' % time.strftime("%Y%m%d-%H%M%S")

# Create folder (if it does not exist)
path.mkdir(parents=True, exist_ok=True)

# Save
fig.write_html("%s/%s" % (path, filename))

# Show
show(fig)
"""