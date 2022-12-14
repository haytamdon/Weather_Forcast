import tensorflow as tf
from tensorflow import keras
from comet_ml import Experiment
import logging

def trainer(model, train, val, params, opt = "rmsprop", loss_func = "mse", metric_list = ["mae"], epoch_num= 50):
    exp=Experiment(project_name="Temperature_Forcast",
                    auto_histogram_gradient_logging=True)
    exp.log_parameters(params)
    callbacks = [
        keras.callbacks.ModelCheckpoint("LSTM_model_best.keras",
                                        save_best_only=True)
    ]
    model.compile(optimizer=opt , loss=loss_func , metrics= metric_list)
    history = model.fit(train,
                    epochs= epoch_num,
                    validation_data=val,
                    callbacks=callbacks)
    return history

