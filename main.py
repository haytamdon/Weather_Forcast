from model.lstm_model import *
from preprocessing.data_preprocessing import *
from trainer.trainer import *
from utils.utils import *

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

if __name__=="__main__":
    X, y = read_data()
    train, val, test = preprocess_data(X, y)
    model = LSTM_model(X)
    history = trainer(model, train, val)
    plot_results(history)