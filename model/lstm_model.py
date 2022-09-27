import tensorflow as tf
from tensorflow import keras
from keras import layers

def LSTM_model(X, sequence_length= 120):
    inputs = keras.Input(shape=(sequence_length, X.shape[-1]))
    x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    return model
