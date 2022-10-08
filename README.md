# Weather_Forcast
This repo is an End-to-End project based on time series forcasting using LSTM and Gated RNN deep learning models to forcast the evolution of temperature based on the Jena Dataset using TensorFlow, and Comet ML for model monitoring and Fast API for deployment

## Usage

Run the 'main.py' script to load the model and to train and to log the main metrics, you can change the model to one of the models existing in the 'models.py' script and rerun the 'main.py' script with one of those models.

## Deployment

run the 'server.py' script in API server folder to execute the server hosting the model's API with FastApi, and run 'fast_api_inference.py' script to call the API and get the model predictions for our inference dataset
## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
