from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from tensorflow import keras
import numpy as np

class Features(BaseModel):
    note: str
    data: List[float]
    

model_file_path = "../../LSTM_model_best.keras"
model = tf.keras.models.load_model(model_file_path)

def process_data(X):
    X = np.Array(X)
    inference = keras.utils.timeseries_dataset_from_array(
        X,
        targets=None,
        sampling_rate= 6,
        sequence_length= 120,
        shuffle=True,
        batch_size=batch_size,
        start_index= 0)
    return inference

app = FastAPI()
@app.post("/run_model")
async def run_model(features: Features):
    print( {
        'note': features.note,
        'list': features.data
        } )
    data = process_data(features.data)
    y_pred = model.predict([X])
    print(Y_pred)
    return {
                "note": features.note,
                "value": Y_pred
            }