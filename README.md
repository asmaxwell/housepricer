# Housepricer

This is a command line tool written in Python to train models for predicting house prices.

# Installation
The dependencies are listed in requirements.txt, and can be automatically installed using:
`pip install -r requirements.txt`

# Basic Usage
After downloading the repo open a terminal in the root housepricer directory. 

Run: `python3 housepricer --data_dir data/ --model_dir data/ --hyperparameter_search random --iterations 150` for an example of setting hyperparameters with a random search

Run: `python3 housepricer data/ data/ --hyperparameter_search evolve --population 50 --generations 25` for an example of setting hyperparameters with an evolutionary algorithm

If you want to run the californian test dataset use the flag `--load_cal True` for example you could run:

`python3 housepricer --model_dir data/ --load_cal True --hyperparameter_search evolve --population 100 --generations 50`

If you want to use the sklearn Histogram-based Gradient Boosting Regression Tree estimator instead of Random forest run

`python3 housepricer --model_dir data/ --model_type notrandom --load_cal True --hyperparameter_search evolve --population 100 --generations 50`

Note currently the included Bristol (UK) dataset is only from the freely available LandRegistry, which does not provide key features such as the number of rooms, size, or overal state of a property. The sklearn Californian dataset include both number of rooms and area of each property, so can be trained to a higher r2 value.