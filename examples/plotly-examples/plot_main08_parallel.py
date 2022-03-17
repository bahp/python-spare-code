import plotly.graph_objects as go

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


def load_gridsearch_sklearn_iris():
    """This method..."""
    # Define datapath
    FILEPATH = './data/sklearn-gridsearch/ls2d-iris.csv'
    # Load data
    df = pd.read_csv(FILEPATH)
    # Columns
    columns = [
        ('mean_train_spearman', 'Spearman'),
        ('mean_train_pearson', 'Pearson'),
        ('param_sae__max_epochs', 'Max Epochs'),
        ('param_sae__lr', 'Learning Rate'),
        ('mean_train_procrustes', 'Procrustes'),
        ('mean_train_calinski_target', 'Calinski'),
        ('mean_train_davies_b_target', 'Davies'),
        #('param_sae__module__layers', 'Layers')
    ]
    # Line
    line = dict(color=df.mean_train_calinski_target,
       colorscale='Electric',
       showscale=True,
       cmin=df.mean_train_calinski_target.min(),
       cmax=df.mean_train_calinski_target.max())
    # Return
    return df, line, columns



def load_raw_dengue():
    """This method..."""
    # Define datapath
    FILEPATH = './data/dengue/data.csv'
    # Load data
    df = pd.read_csv(FILEPATH)
    # Columns
    columns = [
        ('age', 'Age'),
        ('body_temperature', 'Body Temperature'),
        ('weight', 'Weight'),
        ('plt', 'Platelets'),
        ('haematocrit_percent', 'Haematocrit')
    ]
    # Line
    line = dict(color=df.haematocrit_percent,
        colorscale='Electric',
        showscale=True,
        cmin=df.haematocrit_percent.min(),
        cmax=df.haematocrit_percent.max())
    # Return
    return df, line, columns

def create_dimension(s):
    """This method..."""
    if is_numeric_dtype(s):
        print('numeric', s.name)
        return dict(
            range=[s.min(), s.max()],
            constraintrange=[s.min(), s.max()],
            label=s.name, values=s)
    if is_string_dtype(s):
        s = s.apply(str)
        return dict(tickvals=np.arange(s.nunique()),
            ticktext=sorted(s.unique()),
            label=s.name, values=s)


# Load data
df, line, columns = load_raw_dengue()
#df, columns = load_gridsearch_sklearn_iris()

print(df.columns)

# dict(tickvals = [0,0.5,1,2,3],
#     ticktext = ['A','AB','B','Y','Z'],
#     label = 'Cyclinder Material', values = df['cycMaterial']),

# Show
fig = go.Figure(data=
    go.Parcoords(line=line,
        dimensions=[create_dimension(df[name])
            for name, label in columns]
    ))





"""
        list([
            dict(range = [df.mean_train_spearman.min(),
                          df.mean_train_spearman.max()],
                 constraintrange = [df.mean_train_spearman.min(),
                                    df.mean_train_spearman.max()],
                 label="Spearman", values=df.mean_train_spearman),

            dict(range=[df.mean_train_pearson.min(),
                        df.mean_train_pearson.max()],
                 constraintrange=[df.mean_train_pearson.min(),
                                  df.mean_train_pearson.max()],
                 label="Spearman", values=df.mean_train_pearson),

            ])
            #dict(range = [32000,227900],
            #     constraintrange = [100000,150000],
            #     label = "Block Height", values = df['blockHeight']),
            #dict(range = [0,700000],
            #     label = 'Block Width', values = df['blockWidth']),
            #dict(tickvals = [0,0.5,1,2,3],
            #     ticktext = ['A','AB','B','Y','Z'],
            #     label = 'Cyclinder Material', values = df['cycMaterial']),
            #dict(range = [-1,4],
            #     tickvals = [0,1,2,3],
            #     label = 'Block Material', values = df['blockMaterial']),
            #dict(range = [134,3154],
            #     visible = True,
            #     label = 'Total Weight', values = df['totalWeight']),
            #dict(range = [9,19984],
            #     label = 'Assembly Penalty Wt', values = df['assemblyPW']),
            #dict(range = [49000,568000],
            #     label = 'Height st Width', values = df['HstW'])])
    )
)
"""
fig.show()
