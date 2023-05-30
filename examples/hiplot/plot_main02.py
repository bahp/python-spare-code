"""
02. Creating Hiplot HTML
=========================
"""
# Libraries
import pandas as pd
import hiplot as hip

try:
    __file__
    TERMINAL = True
except:
    TERMINAL = False

# -----------------------------
# Load data
# -----------------------------
# Sample data
data = [
    {'dropout':0.1, 'lr': 0.001, 'loss': 10.0, 'optimizer': 'SGD'},
    {'dropout':0.15, 'lr': 0.01, 'loss': 3.5, 'optimizer': 'Adam'},
    {'dropout':0.3, 'lr': 0.1, 'loss': 4.5, 'optimizer': 'Adam'}
]

# Path
path = './datasets/20210602-172616-completed/outcome.csv'
#path = './datasets/single-results.csv'
path = '../../datasets/gridsearch-workbench-dataset/20210602-134857-completed/outcome.csv'
path = '../../datasets/gridsearch-workbench-dataset/20200820-145219-completed/outcome.csv'

# Data
data = pd.read_csv(path)


print(data)

# ----------------------------
# Columns to keep
# ----------------------------
# constants
DEF_SET = 'hos'
DEF_ALG = 'ann'

# Algorithm selected
#algorithm = data.slug_short.str.endswith('-%s' % DEF_ALG)

# Create params DataFrame
params = pd.json_normalize(data.params.apply(eval))

# Select columns
cols_scores = [c for c in data.columns \
    if c.startswith('mean_%s' % DEF_SET)]
cols_params = [c for c in params.columns \
    if c.startswith('%s__' % DEF_ALG)]
cols_other = ['pipeline', 'mean_fit_time']

# Rename columns
rename = {c: c.replace('mean_%s' % DEF_SET, '')
    for c in cols_scores
}
rename.update({c: c.replace("%s__" % DEF_ALG, '')
    for c in cols_params
})
rename.update({
    'mean_fit_time': '__time',
    'pipeline': '__pipeline'
})

# Format data
data = pd.concat((data, params), axis=1)
data = data[cols_scores +
            cols_params +
            cols_other]
#data = data[algorithm]
data = data.round(decimals=3)
data = data.dropna(axis=1, how='all')
data = data.rename(columns=rename)


# Show data
if TERMINAL:
    print("\nData:")
    print(data)
data

####################################################################
# Now......


# ----------------------------
# Create experiment
# ----------------------------
# Create experiment
#exp = hip.Experiment.from_iterable(data)
exp = hip.Experiment.from_dataframe(data)

# Provide configuration for the parallel plot
exp.display_data(hip.Displays.PARALLEL_PLOT).update({
    # Hide from parallel plot
    'hide': ['_aucroc', '_fn', '_fp', '_tn', '_tp'],
    'order': sorted(data.columns),
    'color_by': ['_gmean']  # Does not work
})

# Provide configuration for the table with all the rows
exp.display_data(hip.Displays.TABLE).update({
    'hide': ['from_uid', '_aucroc'],   # Hide columns
    'order': sorted(data.columns),     # Does not work
    'order_by': [['_gmean', 'desc']],  # Order
})

#exp.parameters_definition["_gmean"].colormap = "interpolateSinebow"

# Force ranges of scores (and therefore colors) between 0 and 1.
# exp.parameters_definition["_gmean"].force_range(0, 1)
# exp.parameters_definition["_sens"].force_range(0, 1)
# exp.parameters_definition["_spec"].force_range(0, 1)
# exp.parameters_definition["_aucroc"].force_range(0, 1)

# ----------------------------
# Export
# ----------------------------
# Create html
html = exp.to_html()

# Show
print(html)

# Export
exp.to_html('test.html')