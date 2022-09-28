from model.lstm_model import *
from preprocessing.data_preprocessing import *
from trainer.trainer import *
from utils.utils import *

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from comet_ml import Experiment
import logging

if __name__=="__main__":
    X, y = read_data()
    params = {
        'sequence_length': 120,
        'input_size': X.shape[-1],
        'LSTM_embeddings': 32,
        'LSTM_recurrent_dropout': 0.25,
        'dropout': 0.5,
        'dense_layer': 1,
        'optimizer': 'rmsprop',
        'loss_function': 'mse',
        'metric': 'mae',
        'num_epochs': 50
    }
    train, val, test = preprocess_data(X, y)
    model = LSTM_model(X)
    history = trainer(model, train, val, params)
    plot_results(history)