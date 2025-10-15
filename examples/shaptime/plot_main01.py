"""
Explaining Time-Series Predictions with TimeSHAP and LSTMs
==========================================================

This script demonstrates how to use the TimeSHAP library to
interpret the predictions of a Long Short-Term Memory (LSTM)
model. It begins by generating a synthetic time-series dataset
for a binary classification task. Subsequently, a simple LSTM
model is built and trained using TensorFlow/Keras. The core of
the script then applies TimeSHAP to generate a local explanation
for a single prediction, calculating the necessary baselines
and producing a detailed visual report. This report breaks down
the model's decision-making process by showing the contribution
of each feature at every timestep.
"""
# Libraries
import numpy as np
import pandas as pd

# --------------------------------------------
# Create data
# --------------------------------------------
# Constants
SAMPLES = 1000
TIMESTEPS = 10
FEATURES = 5

# .. note: Either perform a pre-processing step such as
#          normalization or generate the features within
#          the appropriate interval.
# Create dataset
x = np.random.randint(low=0, high=100,
    size=(SAMPLES, TIMESTEPS, FEATURES))
y = np.random.randint(low=0, high=2, size=SAMPLES).astype(float)
i = np.vstack(np.dstack(np.indices((SAMPLES, TIMESTEPS))))

# Create DataFrame
df = pd.DataFrame(
    data=np.hstack((i, x.reshape((-1,FEATURES)))),
    columns=['id', 't'] + ['f%s'%j for j in range(FEATURES)]
)

# Show
print("Shapes:")
print("i: %s" % str(i.shape))
print("y: %s" % str(y.shape))
print("x: %s" % str(x.shape))

print("\nData (%s):" % str(x.shape))
print(x)

print("\nDataFrame (2D)")
print(df)


# --------------------------------------------
# Model
# --------------------------------------------
# Libraries
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

# Create model
model = Sequential()
model.add(Input(shape=(None, FEATURES)))
model.add(
    LSTM(
        units=64,
        return_sequences=False,
        input_shape=(TIMESTEPS, FEATURES)
    ))
#model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Load pre-trained weights

# Display model summary
print(model.summary())

# Fit
model.fit(x, y, epochs=16, batch_size=64)


# --------------------------------------------
# TIME SHAP
# --------------------------------------------
# Libraries
from timeshap.utils import calc_avg_event
from timeshap.utils import calc_avg_sequence
from timeshap.utils import get_avg_score_with_avg_event

#
features = df.columns[2:].tolist()
sequence_id = 'id'
time = 't'
label = None

# Create entry point
f = lambda x: model.predict(x)

# Base line event
average_event = calc_avg_event(df,
    numerical_feats=features,
    categorical_feats=[])
print(average_event)

# Base line event sequence
average_sequence = calc_avg_sequence(df,
    numerical_feats=features,
    categorical_feats=[],
    model_features=features,
    entity_col=sequence_id)
print(average_sequence)

# Average score over baseline
avg_score_over_len = get_avg_score_with_avg_event(f, average_event, top=5)
print(avg_score_over_len)



#positive_sequence_id = f"cycling_{np.random.choice(ids_for_test)}"
#pos_x_pd = d_test_normalized[d_test_normalized['all_id'] == positive_sequence_id]

positive_sequence_id = 0
pos_x_pd = df[df.id == 0]


# select model features only
pos_x_data = pos_x_pd[features]
# convert the instance to numpy so TimeSHAP receives it
pos_x_data = np.expand_dims(pos_x_data.to_numpy().copy(), axis=0)

from timeshap.explainer import local_report
plot_feats = {
    'f0': "f0",
    'f1': "f1",
    'f2': "f2",
}
pruning_dict = {'tol': 0.025}
event_dict = {'rs': 42, 'nsamples': SAMPLES}
feature_dict = {'rs': 42, 'nsamples': SAMPLES, 'feature_names': features, 'plot_features': plot_feats}
cell_dict = {'rs': 42, 'nsamples': SAMPLES, 'top_x_feats': 2, 'top_x_events': 2}
local_report(f, pos_x_data, pruning_dict, event_dict, feature_dict, 
    cell_dict=cell_dict, entity_uuid=positive_sequence_id, 
    entity_col='id', baseline=average_event)




"""
BATCH_SIZE = 16

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#model = MT_LSTM(
    #    timesteps=timesteps,
    #    features=X_train.shape[2],
    #    outputs=1
    #)

model.compile(
        loss='binary_crossentropy', # f1_loss, binary_crossentropy
        optimizer=optimizer,        # optimizer, adam, ...
        metrics=[
            METRICS.get('acc'),
            METRICS.get('prec'),
            METRICS.get('recall'),
            METRICS.get('auc'),
            METRICS.get('prc'),
            METRICS.get('tp'),
            METRICS.get('tn'),
            METRICS.get('fp'),
            METRICS.get('fn'),
        ]
    )
"""

local_report(f, pos_x_data, pruning_dict, event_dict,
             feature_dict,
             cell_dict=cell_dict,
             entity_uuid=positive_sequence_id,
             entity_col='all_id',
             baseline=average_event)
