"""
Created by Andy S Maxwell 11/02/2024
Class to train, search hyper parameters, and deploy random forest models for Bristol House Price Project
"""

import pandas as pd
import numpy as np
from housepricer import housepricetrainer as hpt
from sklearn import preprocessing
from sklearn import model_selection

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import model_selection

from sklearn_genetic import GASearchCV
from sklearn_genetic import ExponentialAdapter
from sklearn.model_selection import StratifiedKFold
from sklearn_genetic.space import Continuous, Categorical, Integer
import joblib

class random_forest:
    data_directory : str
    model_directory : str
    model_filename : str
    data : list[float]
    features : list[str]
    numerical_features : list[str]
    categorical_features : list[str]
    X : list[float]
    y : list[float]
    X_train : list[float]
    y_train : list[float]
    X_test : list[float]
    y_test : list[float]


    def __init__(self, data_directory: str, model_directory: str, model_filename : str = None)  ->None:
        self.data_directory = data_directory
        self.load_data()

        # select and engineer features
        self.features = ['year', 'month', 'day', 'latitude', 'longitude'
            ,'property_type','new_build', 'estate_type'
            , 'transaction_category']

        # use dummy encoding for categories
        self.numerical_features = ['year', 'month', 'day', 'latitude', 'longitude']
        self.categorical_features = ['property_type', 'new_build','estate_type','transaction_category']

        self.get_XY()
        self.get_test_train_split()

        self.model_filename = model_filename
        if(model_filename != None):
            self.model_directory = model_directory
            self.load_model(model_filename)
        return


    def load_data(self) -> None:
        """
        function to load house price data as member variable
        """
        data = pd.read_csv(self.data_directory + "ppd_data.csv")
        # remove NaN and scale price
        data = data[data.loc[:,'postcode'].notna()]

        # scale price paid in millions
        data.loc[:,'price_paid'] = data.loc[:,'price_paid']/1.0e6

        self.data = data
        return

    def load_model(self, filename: str) -> None:
        """
        If a model has been previously trained and stored it can be loaded as a member variable 
        """
        self.model = joblib.load(self.model_directory + filename)
        self.model_filename = filename
        return

    def save_model(self, filename: str) -> None:
        joblib.dump(self.model, self.model_directory + filename)
        return

    def get_XY(self) -> None:
        """
        Do preprocessing, encoding and produce list for X and Y from Data
        """
        # split features into categorical and numerical
        X_dataframe, y_dataframe = hpt.data_pruning(self.data, self.features)
        
        # keep subset of data for now
        recent_years = X_dataframe['year']>1989 #data starts from around 1990.
        X_dataframe = X_dataframe[recent_years]

        self.y = y_dataframe.loc[recent_years].values.tolist()
        #encode data
        X_dataframe = hpt.data_encoding(X_dataframe, self.categorical_features, self.numerical_features)
        self.X = X_dataframe.values.tolist()
        return
    def get_test_train_split(self) -> None:
        """
        split data into test and train data sets
        """
        if len(self.X)>0 and len(self.y)>0:
            #test/train split
            self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.X, self.y, test_size=0.1, random_state=0)
        else:
            print("Error X and Y not filled yet!")
        return







