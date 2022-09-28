import requests
import pandas as pd
import numpy as np
import json

from preprocessing.data_preprocessing import *

if __name__=="__main__":
    inference = generate_inference_data()
    inference = list(inference)
    features = {
        "note": "inference API",
        "data": list(inference)
    }
    
    API_URL = "http://127.0.0.1:8000/run_model"
    response = requests.post(API_URL, json=features)
    output = json.loads(response.text)["value"]
    print(output)