"""
Serialization...
=====================


"""

# Libraries
import json
import pandas
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
tf.random.set_seed(7)

# ---------------------------------------------------
# Main example
# ---------------------------------------------------
# This is an example of training and LSM copied from
# the machine learning mastery blog. It is just to
# have a simple working model to work from.

# Path
path = './data/passengers.csv'

# Read data
df = pd.read_csv(path,
    usecols=[1], engine='python')

# Convert to numpy
dataset = df.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

def create_model(look_back=1):
    """"""
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # Return
    return model

# create and fit the LSTM network
model = create_model(look_back)
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

"""
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
#plt.show()
"""


# ---------------------------------------
# Save and load
# ---------------------------------------
# Jsonpickle
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

# Configure jsonpickle
jsonpickle.set_preferred_backend('json')
jsonpickle_numpy.register_handlers()


def to_jsonpickle(obj, filename):
    """"""
    with open(filename, 'w') as f:
        f.write(jsonpickle.encode(obj))


def from_jsonpickle(filename):
    """"""
    with open(filename, 'r') as f:
        obj = jsonpickle.decode(f.read())
    return obj


def to_txt(weights, path):
    """Converts a list of np.arrays to weights.

    .. note: The np.arrays must be 2D or less.

    Parameters
    ----------
    weights: list of np.arrays
        The weights
    path: str
        The path to save the .txt files for each layer.

    Returns
    --------
    """
    # Libraries
    from pathlib import Path
    path = Path(path)

    # Warning if more than 2D

    # Save shapes
    shapes = [e.shape for e in weights]
    with open(path / 'shapes.txt', 'w') as f:
        f.write(str(shapes))

    # Save weights
    for i, e in enumerate(weights):
        name = 'weights_layer_%s.txt' % i
        np.savetxt(path / name, e ) #, fmt='%1.100e')

def from_txt(path):
    """Loads from a folder of txt files

    Parameters
    ----------
    path: str
        The path with the .txt files for each layer.

    Returns
    -------
    """
    import glob
    from pathlib import Path
    path = Path(path)

    # Load shapes
    with open(path / 'shapes.txt', 'r') as f:
        shapes = eval(f.read())

    # Load weights
    weights = []
    for e in glob.glob(str(path / 'weights_*')):
        weights.append(np.loadtxt(e))

    # Reshape
    reshaped = []
    for s,w in zip(shapes, weights):
        reshaped.append(np.reshape(w, s))

    # Return
    return reshaped






# ------------------------------------------------------------
# Example I: Serialization using jsonpickle
# ------------------------------------------------------------
"""Description:

The purpose of this example is to demonstrate that by 
employing jsonpickle, we can store and retrieve the model, 
while ensuring that the predictions for the observations 
remain unchanged.
"""

# Paths
path_output_json_scaler = './output/main01/scaler.json'
path_output_json_model = './output/main01/model.json'

# Create path if it does not exist
Path(path_output_json_scaler) \
    .parent.mkdir(parents=True, exist_ok=True)
Path(path_output_json_model) \
    .parent.mkdir(parents=True, exist_ok=True)

# Save
to_jsonpickle(scaler, path_output_json_scaler)
to_jsonpickle(model, path_output_json_model)

# Load model
jsonpickle_scaler = from_jsonpickle(path_output_json_scaler)
jsonpickle_model = from_jsonpickle(path_output_json_model)

# Predict
jsonpickle_model_pred_test = jsonpickle_model.predict(testX)

# Are the predictions equal?
print("==> [jsonpickle] Are test set predictions equal? %s" %
    np.array_equal(testPredict, jsonpickle_model_pred_test))





# ------------------------------------------------------------
# Example II: Serialization using readable txt
# ------------------------------------------------------------
"""Description:

The purpose of this example is to demonstrate that by 
employing a manual approach with readable txt files, we 
can store and retrieve the model, while ensuring that 
the predictions for the observations remain unchanged.
"""

# Define path
path_output_txt = './output/main01/txt'

# Create path if it does not exist
Path(path_output_txt).mkdir(parents=True, exist_ok=True)

# Get weights from model
w = jsonpickle_model.get_weights()

# Save to txt
to_txt(w, path=path_output_txt)

# Load from txt
weights = from_txt(path=path_output_txt)

# Compare
#for w1, a1 in zip(w, weights):
#    print(np.array_equal(w1, a1))

# Create model again from txt weights
txt_model = create_model(look_back=1)
txt_model.set_weights(weights)
txt_model_pred_test = txt_model.predict(testX)

print(txt_model_pred_test)

# Are the predictions equal?
print("==> [manualtxt] Are test set predictions equal? %s" %
    np.array_equal(testPredict, txt_model_pred_test))



# ---------------------------------------
# Example III: Using skljson
# ---------------------------------------
# It seems that it does not work for the scalers, only for a handful
# of models so might need to do it manually too.

"""
import sklearn_json as skljson
skljson.to_json(new_scaler, './output/main01/test.json')

des_model = skljson.from_json('./output/main01/test.json')
print(des_model)
"""