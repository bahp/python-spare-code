import numpy as np

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras.layers import Embedding, Dropout, Bidirectional, TimeDistributed

#from keras.layers import CuDNNGRU, CuDNNLSTM, Conv1D
#from keras.layers import BatchNormalization, GlobalMaxPooling1D
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
#from tensorflow import set_random_seed


#import tensorflow
#tensorflow.random.set_seed(x)

from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.compat.v1.keras.layers import Conv1D

# In[16]:


x = [
	[
		[1, 1, 3, 5],
		[9, 5, 3, 7],
		[6, 2, 3, 8],
	],
	[
		[1, 1, 3, 5, 5],
		[9, 5, 3, 7, 3],
		[6, 2, 3, 8, 8],
	],
	[
		[1, 1, 3],
		[9, 5, 3],
		[6, 2, 3],
	],
	[
		[4, 1, 1, 3],
		[6, 9, 5, 3],
		[5, 6, 2, 3],
	],
]



x_cate = [[1,2], [3,1], [2,2], [1,2]]



y = [0, 1, 1, 0]




SEED = 1

MAX_TIME_LENGTH = 10
MAX_TIMESERIES_TYPE = 3




# change input format to make it the same shape

data = np.zeros((len(x), MAX_TIMESERIES_TYPE, MAX_TIME_LENGTH), dtype='int32')
i_data = np.zeros((len(x), MAX_TIMESERIES_TYPE, len(x_cate[0])), dtype='int32')

for i, patient in enumerate(x):
    for j, types in enumerate(patient):
        seq_data = pad_sequences(types, maxlen=MAX_TIME_LENGTH)
        data[i, j] = seq_data
        i_data[i, :len(i_inputs[i])] = i_inputs[i]

print(i_data)

def create_model():
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                input_length=MAX_POST_LENGTH,
                                weights=[embedding_matrix],
                                trainable=False)

    sequence_input = Input(shape=(MAX_POST_LENGTH,))
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm_sent = Bidirectional(CuDNNGRU(50, return_sequences=True))(embedded_sequences)
    l_lstm_sent = Dropout(0.2)(l_lstm_sent)
    l_lstm_sent = AttentionWithContext()(l_lstm_sent)
    l_lstm_sent = Dropout(0.2)(l_lstm_sent)
    preds = Dense(units=2, activation='softmax')(l_lstm_sent)
    sentEncoder = Model(sequence_input, preds)
    print(sentEncoder.summary())

    ana_input = Input(shape=(MAX_POSTS, len(i_data[0][0])))

    review_input = Input(shape=(MAX_POSTS, MAX_POST_LENGTH))
    l_lstm_sent = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = concatenate([l_lstm_sent, ana_input])  # combine time series and categories
    l_lstm_sent = BatchNormalization()(l_lstm_sent)
    l_lstm_sent = Dropout(0.2)(l_lstm_sent)
    l_lstm_sent = Bidirectional(CuDNNGRU(16, return_sequences=True))(l_lstm_sent)
    l_lstm_sent = Dropout(0.2)(l_lstm_sent)
    l_lstm_sent = AttentionWithContext()(l_lstm_sent)
    l_lstm_sent = Dropout(0.2)(l_lstm_sent)
    preds = Dense(2, activation='softmax')(l_lstm_sent)
    model = Model([review_input, ana_input], preds)
    print(model.summary())

    from keras.optimizers import Adam, AdaMod

    adam = AdaMod()
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])

    return model




model = create_model()

model.fit([data, data_i], np.asarray(y, 'int32'),
          shuffle=False,
          nb_epoch=200, batch_size=32, verbose=0)

