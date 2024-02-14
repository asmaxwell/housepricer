"""
Created by Andy S Maxwell 11/02/2024
Class to train, search hyper parameters, and deploy random forest models for Bristol House Price Project
"""

import pandas as pd
import numpy as np
from . import data_wrangling as hpt
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
    postcode_directory : str
    model_filename : str
    model : RandomForestRegressor
    model_scaler : preprocessing.MinMaxScaler
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


    def __init__(self, data_directory: str
                 , model_directory: str
                 , model_filename : str = None
                 , postcode_directory : str = None)  ->None:
        self.data_directory = data_directory
        self.postcode_directory = postcode_directory
        self.load_data()

        # select and engineer features
        self.features = ['year', 'month', 'day', 'latitude', 'longitude'
            ,'property_type','new_build', 'estate_type'
            , 'transaction_category']

        # use dummy encoding for categories
        self.numerical_features = ['year', 'month', 'day', 'latitude', 'longitude']
        self.categorical_features = ['property_type', 'new_build','estate_type','transaction_category']

        self.get_XY()
        self.get_test_train_split(0.1)

        self.model_filename = model_filename
        self.model_directory = model_directory
        self.model_scaler = preprocessing.MinMaxScaler()
        if model_filename != None:
            self.load_model(model_filename)
        else: #initalize default random forest model
            self.model = make_pipeline(self.model_scaler, RandomForestRegressor(n_jobs = -1, random_state=0))

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
        X_dataframe, y_dataframe = hpt.data_pruning(self.data, self.features) if self.postcode_directory == None else hpt.data_pruning(self.data, self.features, self.postcode_directory) 
        
        # keep subset of data for now
        recent_years = X_dataframe['year']>1989 #data starts from around 1990.
        X_dataframe = X_dataframe[recent_years]

        self.y = y_dataframe.loc[recent_years].values.tolist()
        #encode data
        X_dataframe = hpt.data_encoding(X_dataframe, self.categorical_features, self.numerical_features)
        self.X = X_dataframe.values.tolist()
        return
    def get_test_train_split(self, test_size: float = 0.1) -> None:
        """
        split data into test and train data sets
        """
        if len(self.X)>0 and len(self.y)>0:
            #test/train split
            self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.X, self.y, test_size=test_size)
        else:
            print("Error X and Y not filled yet!")
        return
    def set_model_params(self, model_params : dict) -> None:
        """Pass dictionary of parameters to set the random forest model to use
            Note this is setup to be usable with output from RandomizedSearchCV and GASearchCV
            so the dictionary key should all be preprended with randomforestregressor, so to set n_estimators use randomforestregressor__n_estimators as a key
        """
        self.model.set_params(**model_params)
        return
    

    def random_hyperparameter_search(self, iterations : int, param_grid : dict =
                {"randomforestregressor__max_depth": range(5, 101)
               #,"randomforestregressor__bootstrap": [True, False] # can't set max_samples if setting bootstrap
                ,"randomforestregressor__max_features": np.linspace(0.05, 0.95, 40).tolist() +[i for i in range(1, 5)]
                ,"randomforestregressor__n_estimators" : range(50, 2001, 50)
                ,"randomforestregressor__min_samples_split": range(2, 100, 5)
                ,"randomforestregressor__min_samples_leaf": range(1, 15)
                ,"randomforestregressor__max_leaf_nodes": [ i for i in range(2, 30)] + [None]
                ,"randomforestregressor__max_samples": [ i for i in np.linspace(0.05, 1.0, 50)] + [None]
                } 
                , n_jobs : int = 8) -> None:
        """
        Search over grid given to find best model using random iterations
        """
        regr = make_pipeline(self.model_scaler, RandomForestRegressor(random_state=0))
        rf_random = model_selection.RandomizedSearchCV(estimator = regr, param_distributions = param_grid
                                , scoring = 'r2', random_state=0, n_iter = iterations, cv = 5, verbose=0, n_jobs = n_jobs
                                #, error_score='raise'
                                )
        
        rf_random.fit(self.X_train, self.y_train)
        print(f"Best score: {rf_random.best_score_}")
        self.model = rf_random.best_estimator_
        return
    
    def evolve_hyperparameter_search(self, population_size : int, generations : int
                , param_grid : dict =
                {"randomforestregressor__max_depth": Integer(5, 101)
                ,"randomforestregressor__max_features": Continuous(0.01, 0.999, distribution='log-uniform')
                ,"randomforestregressor__n_estimators" : Integer(50, 2000)
                ,"randomforestregressor__min_samples_split": Integer(2, 100)
                ,"randomforestregressor__min_samples_leaf": Integer(1, 15)
                ,"randomforestregressor__max_leaf_nodes": Integer(2, 30)
                ,"randomforestregressor__max_samples": Continuous(0.05, 1.0, distribution='uniform')
                }  
                , n_jobs : int = 8) -> None:
        """
        Search over grid given to find best model using evolution algorithm
        """
        regr = make_pipeline(self.model_scaler, RandomForestRegressor(random_state=0))
        mutation_adapter = ExponentialAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)
        crossover_adapter = ExponentialAdapter(initial_value=0.2, end_value=0.8, adaptive_rate=0.1)
        evolved_estimator = GASearchCV(estimator=regr,
                               cv=5,
                               scoring='r2',
                               population_size=population_size,
                               generations=generations,
                               mutation_probability=mutation_adapter,
                               crossover_probability=crossover_adapter,
                               param_grid=param_grid,
                               n_jobs=n_jobs)
        
        evolved_estimator.fit(self.X_train, self.y_train)
        print(f"Best score: {evolved_estimator.best_score_}")
        self.model = evolved_estimator.best_estimator_
        return
    
    def predict_values(self, X_vals : list[float]) -> float:
        X_scaled = self.model_scaler.transform(X_vals)
        y_scaled = self.model.predict(X_scaled)
        return self.model_scaler.inverse_transform(y_scaled)

    







