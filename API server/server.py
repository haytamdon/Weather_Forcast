from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from tensorflow import keras

class Features(BaseModel):
    note: str
    data: List[float]
    

model_file_path = "../../LSTM_model_best.keras"
model = tf.keras.models.load_model(model_file_path)


app = FastAPI()
@app.post("/run_model")
async def run_model(features: Features):
    print( {
        'note': features.note,
        'list': features.data
        } )
    y_pred = model.predict([features.data])
    print(Y_pred)
    return {
                "note": features.note,
                "value": Y_pred[0, 0]
            }