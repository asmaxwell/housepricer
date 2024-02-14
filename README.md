# Housepricer

This is a command line tool written in Python to train models for predicting house prices. It is currently a work in progress.

# Basic Usage
After downloading the repo open a terminal in the root housepricer directory. 

Run: `python3 housepricer data/ data/ --hyperparameter_search random --iterations 150` for an example of setting hyperparameters with a random search

Run: `python3 housepricer data/ data/ --hyperparameter_search evolve --population 50 --generations 25` for an example of setting hyperparameters with an evolutionary algorithm

Note currently the dataset is only from the freely available LandRegistry, which does not provide key features such as the number of rooms, size, or overal state of a property.