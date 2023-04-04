"""
Shap - Main 08
==============

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
import shap

# Create random training values.
#
# train_x is [
#   [
#        [0.3, 0.54 ... 0.8],
#        [0.4, 0.6 ... 0.55],
#        ...
#   ],
#   [
#        [0.3, 0.54 ... 0.8],
#        [0.4, 0.6 ... 0.55],
#        ...
#   ],
#   ...
# ]
#
# train_y is corresponding classification of train_x sequences, always 0 or 1
# [0, 1, 0, 1, 0, ... 0]
"""
SAMPLES_CNT = 1000

train_x = np.random.rand(SAMPLES_CNT,5,4)
train_y = np.vectorize(lambda x: int(round(x)))(np.random.rand(SAMPLES_CNT))

val_x = np.random.rand(int(SAMPLES_CNT * 0.1),5,4)
val_y = np.vectorize(lambda x: int(round(x)))(np.random.rand(int(SAMPLES_CNT * 0.1)))

# Train model

model = Sequential()
model.add(LSTM(32,input_shape=train_x.shape[1:], return_sequences=False, stateful=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='binary_crossentropy',metrics=['accuracy'])

fit = model.fit(train_x, train_y, batch_size=64, epochs=2,
                validation_data=(val_x, val_y), shuffle=False)

explainer = shap.DeepExplainer(model, train_x[:10])
shap_vals = explainer.shap_values(val_x[:10][:, 0, :])
"""