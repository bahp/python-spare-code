"""
02. Saving & Loading keras models
==================================

This script provides a hands-on demonstration of training an LSTM neural network
for time-series forecasting on the classic airline passenger dataset.

Its primary focus is on illustrating three distinct methods for model
serialization (saving), each suited for a different use case:

1.  **Standard Method (`.keras`):** Simplest approach which saves a
    complete model in a single binary file.
2.  **Architecture + Weights (`.json` / `.h5`):** Separates the model's
    human-readable structure from its binary weights, offering more flexibility.
3.  **Readable Text Export (`.txt`):** A manual method for maximum portability,
    ideal for secure environments or re-implementing the model in another language.

.. warning:: The script provides a mechanism for persisting the corresponding
             scikit-learn scaler; however, this feature has not been formally
             validated. Its performance on sophisticated scikit-learn preprocessing
             pipelines has not been determined. The ability to reuse the identical
             pipeline is a mandatory requirement for achieving a fully reproducible
             prediction workflow
"""

# Libraries
import os
import glob
import numpy as np
import pandas as pd # pandas is needed for reading the URL
import tensorflow as tf
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Input
from sklearn.preprocessing import MinMaxScaler

# fix random seed for reproducibility
tf.random.set_seed(7)

# Configure jsonpickle to handle NumPy arrays
jsonpickle.set_preferred_backend('json')
jsonpickle_numpy.register_handlers()

# -----------------------------------------------------------------
# Helper methods
# -----------------------------------------------------------------
def to_jsonpickle(obj, filename):
    with open(filename, 'w') as f:
        f.write(jsonpickle.encode(obj))

def from_jsonpickle(filename):
    with open(filename, 'r') as f:
        return jsonpickle.decode(f.read())



#%%
# Let's load and prepare the data

# -------------------------------------
# Load and prepare data
# --------------------------------------
print("Loading and preparing data from online URL...")

# URL for the raw airline passengers CSV file
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, usecols=[1], engine='python')
dataset = df.values.astype('float32')

# Normalize the dataset to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Split into train and test sets
train_size = int(len(scaled_data) * 0.67)
train_data = scaled_data[0:train_size]
test_data = scaled_data[train_size:]

# Prepare datasets for training and prediction
look_back = 1
batch_size = 1

# Use the efficient Keras utility for the training dataset
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=train_data[:-look_back],
    targets=train_data[look_back:],
    sequence_length=look_back,
    batch_size=batch_size,
)

# For verification, we need a consistent NumPy array for test predictions
def create_numpy_sequences(data, look_back=1):
    """Creates X and Y NumPy arrays from time series data."""
    X, Y = [], []
    for i in range(len(data) - look_back -1):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

testX, testY = create_numpy_sequences(test_data, look_back)
# Reshape input to be [samples, time steps, features] for the LSTM model
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
print(f"Data prepared. Test input shape: {testX.shape}")


#%%
# Lets create and train our LSTM model

# ---------------------------------------
# Model training
# ---------------------------------------

def create_model(look_back=1):
    """Creates a compiled LSTM model."""
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(4),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Create and train the model
print("\nTraining LSTM model...")
model = create_model(look_back)
model.fit(train_ds, epochs=100, verbose=0)
print("Model training complete.")

# Predictions from the original, in-memory model
original_predictions = model.predict(testX)
print(f"Made baseline predictions. Shape: {original_predictions.shape}")





# %%
# Let's save in memory using standard tensorflow/keras serialization

# --------------------------------------------------
# Example I: Standard Tensorflow/Keras serialization
# --------------------------------------------------
print("\n--- Example I: Standard Keras .save() / .load_model() ---")
path_output_keras = Path('./outputs/main02/keras_model')
keras_model_path = path_output_keras / 'model.keras'
path_output_keras.mkdir(parents=True, exist_ok=True)

# Save the model in the recommended .keras format
model.save(keras_model_path)
print(f"Model saved to {keras_model_path}")

# Load the model back
loaded_keras_model = tf.keras.models.load_model(keras_model_path)
print("Model loaded successfully.")

# Verify predictions
keras_predictions = loaded_keras_model.predict(testX)
are_equal_keras = np.array_equal(original_predictions, keras_predictions)
print(f"==> [Keras] Are test set predictions equal? {are_equal_keras}")




#%%
# Let's save in memory using keras .json + weights

# -----------------------------------------------------
# Example II: Serialisation using keras JSON + weights
# -----------------------------------------------------
print("\n--- Example II: Serialization using Keras JSON (architecture) + H5 (weights) ---")

from tensorflow.keras.models import model_from_json

path_output_jsonp = Path('./outputs/main02/jsonpickle_model')
path_output_arch = path_output_jsonp / 'model_architecture.json'
path_output_weights = path_output_jsonp / 'model_weights.weights.h5' # Use .h5 for weights
path_output_json_scaler = path_output_jsonp / 'scaler.json'
path_output_jsonp.mkdir(parents=True, exist_ok=True)

# --- SAVING ---
# 1. Save the model's architecture as a JSON string
model_json = model.to_json()
with open(path_output_arch, "w") as json_file:
    json_file.write(model_json)

# 2. Save the model's weights
model.save_weights(path_output_weights)

# 3. Save the scaler (jsonpickle is fine for this simple object)
to_jsonpickle(scaler, path_output_json_scaler)
print("Model architecture, weights, and scaler saved successfully.")

# --- LOADING ---
# 1. Load the model architecture from the JSON file
with open(path_output_arch, 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# 2. Load the weights into the new model structure
loaded_model.load_weights(path_output_weights)
print("Model loaded from architecture and weights.")

# 3. CRUCIAL: Compile the model after loading
# The model must be compiled before you can use it.
loaded_model.compile(loss='mean_squared_error', optimizer='adam')

# --- VERIFYING ---
# Now the loaded_model is a proper Keras model and has the .predict method
json_predictions = loaded_model.predict(testX)
are_equal_json = np.array_equal(original_predictions, json_predictions)
print(f"==> [Keras JSON] Are test set predictions equal? {are_equal_json}")




#%%
# Let's save in memory using readable .txt files.

# -----------------------------------------------------
# Example III: Serialization using readable .txt files
# -----------------------------------------------------
print("\n--- Example III: Serialization using readable .txt files ---")

def to_txt(weights, path):
    """Saves model weights and their shapes to a directory of .txt files."""
    path = Path(path)
    # Save shapes
    shapes = [e.shape for e in weights]
    with open(path / 'shapes.txt', 'w') as f:
        f.write(str(shapes))
    # Save weights
    for i, e in enumerate(weights):
        # Flatten array to save, will be reshaped on load
        np.savetxt(path / f'weights_layer_{i}.txt', e.flatten())

def from_txt(path):
    """Loads weights from a directory of .txt files."""
    path = Path(path)
    # Load shapes from shapes.txt
    with open(path / 'shapes.txt', 'r') as f:
        shapes = eval(f.read())
    # Load weights and reshape
    weights = []
    for i, shape in enumerate(shapes):
        w = np.loadtxt(path / f'weights_layer_{i}.txt')
        weights.append(w.reshape(shape))
    return weights

# Create folder
path_output_txt = Path('./outputs/main02/txt_model')
path_output_txt.mkdir(parents=True, exist_ok=True)

# Get weights from the original model and save them
original_weights = model.get_weights()
to_txt(original_weights, path=path_output_txt)
print(f"Model weights saved to text files in {path_output_txt}")

# Load the weights from the text files
loaded_txt_weights = from_txt(path=path_output_txt)
print("Weights loaded successfully from text files.")

# To use these weights, we must create an identical model structure
txt_model = create_model(look_back)
txt_model.set_weights(loaded_txt_weights)
print("New model created and weights have been set.")

# Verify predictions
txt_predictions = txt_model.predict(testX)
are_equal_txt = np.array_equal(original_predictions, txt_predictions)
print(f"==> [manual .txt] Are test set predictions equal? {are_equal_txt}")