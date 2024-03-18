"""
Created by Andy S Maxwell 11/02/2024
Class to train, search hyper parameters, and deploy random forest models for Bristol House Price Project
"""

from housepricer.debug import silence_tensorflow
#tensorflow is not actually used, it is just imported by the sklearn genetic package
#The information and warning messages are suppressed to not confuse the user
silence_tensorflow()
import numpy as np
from abc import ABCMeta
from . import data_wrangling as hpt
from . import plot
from sklearn import preprocessing
from sklearn.base import BaseEstimator

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn import model_selection

from sklearn_genetic import GASearchCV
from sklearn_genetic import ExponentialAdapter
from sklearn.metrics import mean_squared_error
from sklearn_genetic.space import Continuous, Integer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
#import joblib
import dill

class trainer:
    wrangler : hpt.wrangling
    model_directory : str
    model_args : dict
    model_filename : str
    model_type : ABCMeta
    model : BaseEstimator
    model_scaler : preprocessing.MinMaxScaler
    model_searcher : BaseEstimator
    X_train : list[float]
    y_train : list[float]
    X_test : list[float]
    y_test : list[float]


    def __init__(self, data_directory: str
                 , model_directory: str
                 , model_filename : str = None
                 , postcode_directory : str = None
                 , load_cal_data : bool = False
                 , model_type : str = "random"
                 )  ->None:
        """
            Init function, loading data, postcodes, and model if supplied. Test californian data can be loaded if specified
        """
        self.wrangler = hpt.wrangling(data_directory, postcode_directory, load_cal_data)
        self.get_test_train_split(0.1)

        self.model_filename = model_filename
        self.model_directory = model_directory
        self.model_scaler = preprocessing.MinMaxScaler()
        self.model_searcher = None
        if model_type == "random":
            self.model_type = RandomForestRegressor
            self.model_args = {
                "random_state" : 0
            }
        else:
            self.model_type = HistGradientBoostingRegressor
            self.model_args = {
                "random_state" : 0
                ,"early_stopping" : False
                #,"categorical_features" : 
            }

        if model_filename != None:
            self.load_model(model_filename)
        else: #initalize model
            self.model = make_pipeline(self.model_scaler, self.model_type(**self.model_args))

        return
    def get_test_train_split(self, test_size: float = 0.1) -> tuple[list]:
        """
        split data into test and train data sets using wrangling class
        """
        tupleOut = self.wrangler.get_test_train_split(test_size)
        if(len(tupleOut)>0):
            self.X_train, self.X_test, self.y_train, self.y_test = tupleOut
        else:
            print("Error Test Train split not performed!")


    def load_model(self, filename: str) -> None:
        """
        If a model has been previously trained and stored it can be loaded as a member variable 
        """
        with open(self.model_directory + filename, 'rb') as pickle_file:
            self.model_searcher = dill.load(pickle_file, ignore=False)

        self.model = self.model_searcher.best_estimator_
        self.model_filename = filename
        return

    def save_model(self, filename: str) -> None:
        pickles = dill.pickles(self.model_searcher, exact=False, safe=False)
        print(f"Does it pickle: {pickles}")
        with open(self.model_directory + filename, 'wb') as pickle_file:
            dill.dump(self.model_searcher, pickle_file)
            #dill.dump_module(filename = self.model_directory + filename, module = self.model_searcher)
        return
    
    def set_model_params(self, model_params : dict) -> None:
        """Pass dictionary of parameters to set the random forest model to use
            Note this is setup to be usable with output from RandomizedSearchCV and GASearchCV
            so the dictionary key should all be preprended with randomforestregressor, so to set n_estimators use randomforestregressor__n_estimators as a key
        """
        self.model.set_params(**model_params)
        return    

    def random_hyperparameter_search(self, iterations : int
                , n_jobs : int = 8) -> None:
        """
        Search over grid given to find best model using random iterations
        """
        if self.model_type == RandomForestRegressor:
            param_grid = {"randomforestregressor__max_depth": range(5, 101)
                        #,"randomforestregressor__bootstrap": [True, False] # can't set max_samples if setting bootstrap
                            ,"randomforestregressor__max_features": np.linspace(0.05, 0.95, 40).tolist() +[i for i in range(1, 5)]
                            ,"randomforestregressor__n_estimators" : range(50, 2001, 50)
                            ,"randomforestregressor__min_samples_split": range(2, 100, 5)
                            ,"randomforestregressor__min_samples_leaf": range(1, 15)
                            ,"randomforestregressor__max_leaf_nodes": [ i for i in range(2, 30)] + [None]
                            ,"randomforestregressor__max_samples": [ i for i in np.linspace(0.05, 1.0, 50)] + [None]
            }  
        else:
            param_grid = {"histgradientboostingregressor__max_depth": [i for i in range(5, 100, 2)] + [None]
                        #,"histgradientboostingregressor__max_features": np.linspace(0.05, 0.95, 40).tolist() +[i for i in range(1, 5)]
                        ,"histgradientboostingregressor__max_iter" : range(5, 500, 10)
                        ,"histgradientboostingregressor__min_samples_leaf": range(1, 50, 2)
                        ,"histgradientboostingregressor__max_leaf_nodes": [ i for i in range(2, 100, 2)]
            }
        regr = make_pipeline(self.model_scaler, self.model_type(**self.model_args) )
        rf_random = model_selection.RandomizedSearchCV(estimator = regr, param_distributions = param_grid
                                , scoring = 'r2', random_state=0, n_iter = iterations, cv = 5, verbose=0, n_jobs = n_jobs
                                #, error_score='raise'
                                )
        
        rf_random.fit(self.X_train, self.y_train)
        self.model_searcher = rf_random
        print(f"Best score: {rf_random.best_score_}")
        self.model = rf_random.best_estimator_
        print(f"Test Error: {self.test_error()}")
        return
    
    def evolve_hyperparameter_search(self, population_size : int, generations : int
                , n_jobs : int = -1) -> None:
        """
        Search over grid given to find best model using evolution algorithm
        """
        if self.model_type == RandomForestRegressor:
            param_grid = {
                    "randomforestregressor__max_depth": Integer(5, 101)
                    ,"randomforestregressor__max_features": Continuous(0.01, 0.999, distribution='log-uniform')
                    ,"randomforestregressor__n_estimators" : Integer(50, 2000)
                    ,"randomforestregressor__min_samples_split": Integer(2, 100)
                    ,"randomforestregressor__min_samples_leaf": Integer(1, 15)
                    ,"randomforestregressor__max_leaf_nodes": Integer(2, 30)
                    ,"randomforestregressor__max_samples": Continuous(0.05, 1.0, distribution='uniform')
            }
        else:
            param_grid = {
                    "histgradientboostingregressor__max_depth": Integer(5, 100)
                    #,"histgradientboostingregressor__max_features": np.linspace(0.05, 0.95, 40).tolist() +[i for i in range(1, 5)]
                    ,"histgradientboostingregressor__max_iter" : Integer(5, 500)
                    ,"histgradientboostingregressor__min_samples_leaf": Integer(1, 50)
                    ,"histgradientboostingregressor__max_leaf_nodes": Integer(2, 100)
            }        

        regr = make_pipeline(self.model_scaler, self.model_type(**self.model_args))
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
        self.model_searcher = evolved_estimator
        print(f"Best score: {evolved_estimator.best_score_}")
        self.model = evolved_estimator.best_estimator_
        #self.model_scaler = evolved_estimator.sca
        print(f"Average Train Error: {self.train_error()}")
        print(f"Average Test Error: {self.test_error()}")
        return
    
    def is_model_fitted(self)->None:
        some_test_data = [[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2021.0, 8.0, 10.0, 367118.0, 180928.0]]
        try:
            print(self.model_searcher)
            #X_scaled = self.model_searcher.transform(some_test_data)
            #self.model.predict(X_scaled)
            print(self.model.predict(some_test_data))
        except NotFittedError as exc:
            print(repr(exc))
            raise
        return
        
    def predict_values(self, X_vals : list[float]) -> float:
        try:
            check_is_fitted(self.model)
        except NotFittedError as exc:
            print(f"Model is not fitted yet.")
        #X_scaled = self.model_searcher.transform(X_vals)
        y_scaled = self.model.predict([X_vals])
        return y_scaled * 1.0e+06 #self.model_searcher.inverse_transform(y_scaled)
    
    def test_error(self) -> float:
        y_pred = self.model.predict (self.X_test)
        return mean_squared_error(self.y_test, y_pred)#/np.mean(self.y_test)
    def train_error(self) -> float:
        y_pred = self.model.predict (self.X_train)
        return mean_squared_error(self.y_train, y_pred)#/np.mean(self.y_train)
    
    def save_true_vs_predicted(self, filename : str):
        """
        Save to output directory a figure of true vs predicted value for the training and test sets
        """
        y_pred = model_selection.cross_val_predict(self.model, self.X_train, self.y_train, cv=5)
        #hpt.plot_cross_validated_pred(np.log(self.y_train), np.log(y_pred), self.model_directory + "train_" + filename )
        plot.plot_cross_validated_pred(self.y_train, y_pred, self.model_directory + "train_" + filename )

        y_pred = model_selection.cross_val_predict(self.model, self.X_test, self.y_test, cv=5)
        #hpt.plot_cross_validated_pred(np.log(self.y_test), np.log(y_pred), self.model_directory + "test_" + filename  )
        plot.plot_cross_validated_pred(self.y_test, y_pred, self.model_directory + "test_" + filename  )



    







