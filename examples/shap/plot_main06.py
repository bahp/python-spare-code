# Libraries
import shap
import numpy as np
import pandas as pd

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

# --------------------------------------------
# Create data
# --------------------------------------------
# Constants
SAMPLES = 10
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
#model.add(Input(shape=(None, FEATURES)))
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
model.run_eagerly = False

# Load pre-trained weights

# Display model summary
print(model.summary())

model.save('model.h5')

print(x)
print(x[:10])
print("AHH")
# Fit
model.fit(x, y, epochs=16, batch_size=64)



# --------------------------------------------
# Compute and display SHAP values
# --------------------------------------------
# https://github.com/slundberg/shap/blob/master/shap/plots/_beeswarm.py

# Use the training data for deep explainer => can use fewer instances
explainer = shap.DeepExplainer(model, x)
# explain the the testing instances (can use fewer instanaces)
# explaining each prediction requires 2 * background dataset size runs
shap_values = explainer.shap_values(x)
# init the JS visualization code
shap.initjs()

print(shap_values[0].shape)

shap_values = explainer(x)
shap.plots.beeswarm(shap_values,
    max_display=12, order=shap.Explanation.abs.mean(0))

import matplotlib.pyplot as plt
plt.show()




import sys
sys.exit()


shap_values_2D = shap_values[0].reshape(-1,x.shape[-1])
x_2D = pd.DataFrame(
    data=x.reshape(-1,x.shape[-1]),
    columns=['f%s'%j for j in range(x.shape[-1])]
)


## SHAP for each time step
NUM_STEPS = x.shape[1]
NUM_FEATURES = x.shape[-1]
len_test_set = x_2D.shape[0]

"""
# step = 0
for step in range(NUM_STEPS):
    indice = [i for i in list(range(len_test_set)) if i%NUM_STEPS == step]
    shap_values_2D_step = shap_values_2D[indice]
    x_test_2d_step = x_2D.iloc[indice]
    print("_______ time step {} ___________".format(step))
    #shap.summary_plot(shap_values_2D_step, x_test_2d_step, plot_type="bar")
    shap.summary_plot(shap_values_2D_step, x_test_2d_step)
    print("\n")
"""


shap_values_2D_step = shap_values_2D[:, 1].reshape(-1, x.shape[1])
x_test_2d_step = x_2D.iloc[:, 1].to_numpy().reshape(-1, x.shape[1])
x_test_2d_step = pd.DataFrame(
    x_test_2d_step, columns=['timestep %s'%j for j in range(x.shape[1])]
)

print(x_test_2d_step)

shap.summary_plot(shap_values_2D_step, x_test_2d_step)

"""
for step in range(NUM_STEPS):
    indice = [i for i in list(range(len_test_set)) if i%NUM_STEPS == step]
    shap_values_2D_step = shap_values_2D[indice]
    x_test_2d_step = x_2D.iloc[indice]
    print("_______ time step {} ___________".format(step))
    #shap.summary_plot(shap_values_2D_step, x_test_2d_step, plot_type="bar")
    shap.summary_plot(shap_values_2D_step, x_test_2d_step)
    print("\n")
"""
import matplotlib.pyplot as plt
plt.show()

"""
#y_pred = model.predict(x[:3, :, :])
#print(y_pred)

#background = x[np.random.choice(x.shape[0], 10, replace=False)]
masker = shap.maskers.Independent(data=x)
# Get generic explainer
#explainer = shap.KernelExplainer(model, background)
explainer = shap.KernelExplainer(model.predict, x, masker=masker)

# Show kernel type
print("\nKernel type: %s" % type(explainer))

# Get shap values
shap_values = explainer.shap_values(x)

print(shap_values)
"""
