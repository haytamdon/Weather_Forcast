import os
import numpy as np

def read_data():
    fname = os.path.join("jena_climate_2009_2016.csv")

    with open(fname) as f:
        data = f.read()

    lines = data.split("\n")
    header = lines[0].split(",")
    lines = lines[1:]
    temperature = np.zeros((len(lines),))
    raw_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(",")[1:]]
        temperature[i] = values[1]
        raw_data[i, :] = values[:]
        return raw_data, temperature
    

def preprocess_data(X,y, sampling_rate = 6, sequence_length = 120, batch_size = 256):
    num_train_samples = int(0.5 * len(X))
    num_val_samples = int(0.25 * len(X))
    num_test_samples = len(X) - num_train_samples - num_val_samples
    mean = X[:num_train_samples].mean(axis=0)
    X -= mean
    std = X[:num_train_samples].std(axis=0)
    X /= std
    delay = sampling_rate * (sequence_length + 24 - 1)
    train = keras.utils.timeseries_dataset_from_array(
        X[:-delay],
        targets=y[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=0,
        end_index=num_train_samples)

    val = keras.utils.timeseries_dataset_from_array(
        X[:-delay],
        targets=y[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=num_train_samples,
        end_index=num_train_samples + num_val_samples)

    test = keras.utils.timeseries_dataset_from_array(
        X[:-delay],
        targets=y[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=num_train_samples + num_val_samples)
    
    return train, val, test

