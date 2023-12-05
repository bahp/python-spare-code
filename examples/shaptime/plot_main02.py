# Libraries
import os
import re
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# Set seed.
np.random.seed(42)

# ------------------------------------------------
# Load data
# ------------------------------------------------
# Too many for
# changing path 3 times in loops.

data_directories = next(os.walk("./data/AReM"))[1]

all_csvs = []
for folder in data_directories:
    if folder in ['bending1', 'bending2']:
        continue
    folder_csvs = next(os.walk(f"./data/AReM/{folder}"))[2]
    for data_csv in folder_csvs:
        if data_csv == 'dataset8.csv' and folder == 'sitting':
            # this dataset only has 479 instances
            # it is possible to use it, but would require padding logic
            continue
        loaded_data = pd.read_csv(f"./data/AReM/{folder}/{data_csv}", skiprows=4)
        print(f"{folder}/{data_csv} ------ {loaded_data.shape}")

        csv_id = re.findall(r'\d+', data_csv)[0]
        loaded_data['id'] = csv_id
        loaded_data['all_id'] = f"{folder}_{csv_id}"
        loaded_data['activity'] = folder
        all_csvs.append(loaded_data)
all_data = pd.concat(all_csvs)
raw_model_features = ['avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']
all_data.columns = ['timestamp', 'avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23', 'id',
                    'all_id', 'activity']


# Show
print("Data:")
print(all_data)

# choose ids to use for test
ids_for_test = np.random.choice(all_data['id'].unique(), size = 4, replace=False)

d_train =  all_data[~all_data['id'].isin(ids_for_test)]
d_test = all_data[all_data['id'].isin(ids_for_test)]

class NumericalNormalizer:
    def __init__(self, fields: list):
        self.metrics = {}
        self.fields = fields

    def fit(self, df: pd.DataFrame ) -> list:
        means = df[self.fields].mean()
        std = df[self.fields].std()
        for field in self.fields:
            field_mean = means[field]
            field_stddev = std[field]
            self.metrics[field] = {'mean': field_mean, 'std': field_stddev}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transform to zero-mean and unit variance.
        for field in self.fields:
            f_mean = self.metrics[field]['mean']
            f_stddev = self.metrics[field]['std']
            # OUTLIER CLIPPING to [avg-3*std, avg+3*avg]
            df[field] = df[field].apply(lambda x: f_mean - 3 * f_stddev if x < f_mean - 3 * f_stddev else x)
            df[field] = df[field].apply(lambda x: f_mean + 3 * f_stddev if x > f_mean + 3 * f_stddev else x)
            if f_stddev > 1e-5:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: ((x - f_mean)/f_stddev))
            else:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: x * 0)
        return df

#all features are numerical
normalizor = NumericalNormalizer(raw_model_features)
normalizor.fit(d_train)
d_train_normalized = normalizor.transform(d_train)
d_test_normalized = normalizor.transform(d_test)

model_features = [f"p_{x}_normalized" for x in raw_model_features]
time_feat = 'timestamp'
label_feat = 'activity'
sequence_id_feat = 'all_id'

plot_feats = {
    'p_avg_rss12_normalized': "Mean Chest <-> Right Ankle",
    'p_var_rss12_normalized': "STD Chest <-> Right Ankle",
    'p_avg_rss13_normalized': "Mean Chest <-> Left Ankle",
    'p_var_rss13_normalized': "STD Chest <-> Left Ankle",
    'p_avg_rss23_normalized': "Mean Right Ankle <-> Left Ankle",
    'p_var_rss23_normalized': "STD Right Ankle <-> Left Ankle",
}

# possible activities ['cycling', 'lying', 'sitting', 'standing', 'walking']
#Select the activity to predict
chosen_activity = 'cycling'

d_train_normalized['label'] = d_train_normalized['activity'].apply(lambda x: int(x == chosen_activity))
d_test_normalized['label'] = d_test_normalized['activity'].apply(lambda x: int(x == chosen_activity))

def df_to_numpy(df, model_feats, label_feat, group_by_feat, timestamp_Feat):
    sequence_length = len(df[timestamp_Feat].unique())

    data_tensor = np.zeros(
        (len(df[group_by_feat].unique()), sequence_length, len(model_feats)))
    labels_tensor = np.zeros((len(df[group_by_feat].unique()), 1))

    for i, name in enumerate(df[group_by_feat].unique()):
        name_data = df[df[group_by_feat] == name]
        sorted_data = name_data.sort_values(timestamp_Feat)

        data_x = sorted_data[model_feats].values
        labels = sorted_data[label_feat].values
        assert labels.sum() == 0 or labels.sum() == len(labels)
        data_tensor[i, :, :] = data_x
        labels_tensor[i, :] = labels[0]
    return data_tensor, labels_tensor

X_train, y_train = df_to_numpy(d_train_normalized, model_features, 'label', sequence_id_feat, time_feat)

X_test, y_test = df_to_numpy(d_test_normalized, model_features, 'label', sequence_id_feat, time_feat)

import tensorflow as tf

inputs = tf.keras.layers.Input(shape=(None, 6))
lstm1 = tf.keras.layers.LSTM(64)(inputs)
ff1 = tf.keras.layers.Dense(64, activation='relu')(lstm1)
ff2 = tf.keras.layers.Dense(1, activation='sigmoid')(ff1)
model = tf.keras.models.Model(inputs=inputs, outputs=ff2)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(0.001))

model.fit(X_train, y_train, epochs=20, batch_size=55, validation_data=(X_test, y_test))











print(d_train_normalized)


import timeshap

f = lambda x: model.predict(x)

from timeshap.utils import calc_avg_event
average_event = calc_avg_event(
    d_train_normalized,
    numerical_feats=model_features,
    categorical_feats=[])
print(average_event)

from timeshap.utils import calc_avg_sequence
average_sequence = calc_avg_sequence(
    d_train_normalized,
    numerical_feats=model_features,
    categorical_feats=[],
    model_features=model_features,
    entity_col=sequence_id_feat)
print(average_sequence)

from timeshap.utils import get_avg_score_with_avg_event
avg_score_over_len = get_avg_score_with_avg_event(f, average_event, top=480)
print(avg_score_over_len)




positive_sequence_id = f"cycling_{np.random.choice(ids_for_test)}"
pos_x_pd = d_test_normalized[d_test_normalized['all_id'] == positive_sequence_id]

# select model features only
pos_x_data = pos_x_pd[model_features]
# convert the instance to numpy so TimeSHAP receives it
pos_x_data = np.expand_dims(pos_x_data.to_numpy().copy(), axis=0)

from timeshap.explainer import local_report

pruning_dict = {'tol': 0.025}
event_dict = {'rs': 42, 'nsamples': 32000}
feature_dict = {'rs': 42, 'nsamples': 32000, 'feature_names': model_features, 'plot_features': plot_feats}
cell_dict = {'rs': 42, 'nsamples': 32000, 'top_x_feats': 2, 'top_x_events': 2}
plot_1 = local_report(f, pos_x_data, pruning_dict, event_dict, feature_dict, cell_dict=cell_dict, entity_uuid=positive_sequence_id, entity_col='all_id', baseline=average_event)

plot_1.save('./outputs/plot1.html')



from timeshap.explainer import global_report

pos_dataset = d_test_normalized[d_test_normalized['label'] == 1]
schema = schema = list(pos_dataset.columns)
pruning_dict = {'tol': [0.05, 0.075], 'path': 'outputs/prun_all_tf.csv'}
event_dict = {'path': 'outputs/event_all_tf.csv', 'rs': 42, 'nsamples': 32000}
feature_dict = {'path': 'outputs/feature_all_tf.csv', 'rs': 42, 'nsamples': 32000, 'feature_names': model_features, 'plot_features': plot_feats,}
prun_stats, global_plot = global_report(f, pos_dataset, pruning_dict, event_dict, feature_dict, average_event, model_features, schema, sequence_id_feat, time_feat, )
prun_stats

global_plot.save('./outputs/plot2.html')


from timeshap.plot import plot_temp_coalition_pruning, plot_event_heatmap, plot_feat_barplot, plot_cell_level
from timeshap.explainer import local_pruning, local_event, local_feat, local_cell_level
# select model features only
pos_x_data = pos_x_pd[model_features]
# convert the instance to numpy so TimeSHAP receives it
pos_x_data = np.expand_dims(pos_x_data.to_numpy().copy(), axis=0)


pruning_dict = {'tol': 0.025,}
coal_plot_data, coal_prun_idx = local_pruning(f, pos_x_data, pruning_dict, average_event, positive_sequence_id, sequence_id_feat, False)
# coal_prun_idx is in negative terms
pruning_idx = pos_x_data.shape[1] + coal_prun_idx
pruning_plot = plot_temp_coalition_pruning(coal_plot_data, coal_prun_idx, plot_limit=40)
pruning_plot.save('./outputs/plot3.html')

event_dict = {'rs': 42, 'nsamples': 32000}
event_data = local_event(f, pos_x_data, event_dict, positive_sequence_id, sequence_id_feat, average_event, pruning_idx)
event_plot = plot_event_heatmap(event_data)
event_plot.save('./outputs/plot4.html')

feature_dict = {'rs': 42, 'nsamples': 32000, 'feature_names': model_features, 'plot_features': plot_feats}
feature_data = local_feat(f, pos_x_data, feature_dict, positive_sequence_id, sequence_id_feat, average_event, pruning_idx)
feature_plot = plot_feat_barplot(feature_data, feature_dict.get('top_feats'), feature_dict.get('plot_features'))
feature_plot.save('./outputs/plot5.html')

cell_dict = {'rs': 42, 'nsamples': 32000, 'top_x_events': 3, 'top_x_feats': 3}
cell_data = local_cell_level(f, pos_x_data, cell_dict, event_data, feature_data, positive_sequence_id, sequence_id_feat, average_event, pruning_idx)
feat_names = list(feature_data['Feature'].values)[:-1] # exclude pruned events
cell_plot = plot_cell_level(cell_data, feat_names, feature_dict.get('plot_features'))
cell_plot.save('./outputs/plot6.html')

from timeshap.explainer import prune_all, pruning_statistics, event_explain_all, feat_explain_all
from timeshap.plot import plot_global_event, plot_global_feat

pos_dataset = d_test_normalized[d_test_normalized['label'] == 1]

pruning_dict = {'tol': [0.05, 0.075], 'path': 'outputs/prun_all_tf.csv'}
prun_indexes = prune_all(f, pos_dataset, pruning_dict, average_event, model_features, schema, sequence_id_feat, time_feat)
pruning_stats = pruning_statistics(prun_indexes, pruning_dict.get('tol'))
#pruning_stats.save('outputs/plot7.html')

event_dict = {'path': 'outputs/event_all_tf.csv', 'rs': 42, 'nsamples': 32000}
event_data = event_explain_all(f, pos_dataset, event_dict, prun_indexes, average_event, model_features, schema, sequence_id_feat, time_feat)
event_global_plot = plot_global_event(event_data)
event_global_plot.save('outputs/plot8.html')

feature_dict = {'path': 'outputs/feature_all_tf.csv', 'rs': 42, 'nsamples': 32000, 'feature_names': model_features, 'plot_features': plot_feats, }
feat_data = feat_explain_all(f, pos_dataset, feature_dict, prun_indexes, average_event, model_features, schema, sequence_id_feat, time_feat)
feat_global_plot = plot_global_feat(feat_data, **feature_dict)
feat_global_plot.save('outputs/plot9.html')