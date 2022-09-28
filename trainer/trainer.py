import tensorflow as tf
from tensorflow import keras

def trainer(model, train, val, opt = "rmsprop", loss_func = "mse", metric_list = ["mae"], epoch_num= 50):
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