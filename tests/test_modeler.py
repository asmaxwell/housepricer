"""
Created by Andy S Maxwell 11/02/2024
Test for model class to train random forest model
"""
import pytest
import numpy as np
from housepricer import modeler
from sklearn_genetic.space import Continuous, Integer
import os


### setup - load database
@pytest.fixture(scope='session') 
def load_database() -> None:
    rf = modeler.trainer("data/", "./")
    yield rf
    return
@pytest.fixture(scope='session') 
def load_cal_database() -> None:
    rf = modeler.trainer("data/", "data/", None, None, True)
    yield rf
    return

@pytest.fixture(scope='session') 
def hist_model() -> None:
    rf = modeler.trainer("data/", "data/", None, None, False, "notrandom")
    yield rf
    return

### test __init__
def test__init__(load_database) -> None:
    """
    This test also tests .load_data() and .load_model()
    """
    rf = load_database
    assert(rf.model_filename == None or rf.model != None)

def test_set_model_params(load_database) -> None:
    rf = load_database
    params = {"randomforestregressor__max_depth": 21
                ,"randomforestregressor__max_features": 13
                ,"randomforestregressor__n_estimators" : 57
                }
    rf.set_model_params(params)
    assert(rf.model.get_params()["randomforestregressor__max_depth"] == 21) 
    assert(rf.model.get_params()["randomforestregressor__max_features"] == 13) 
    assert(rf.model.get_params()["randomforestregressor__n_estimators"] == 57)

def test_set_hist_model(hist_model):
    """
    Test init forHist Model
    """
    rf = hist_model
    assert(rf.model_filename == None or rf.model != None)


@pytest.mark.slow
def test_random_hyperparameter_search(load_database) -> None:
    rf = load_database
    rf.random_hyperparameter_search(3)
    assert(rf.model.get_params()["randomforestregressor__n_estimators"] != 100) #checking if default value no longer active

@pytest.mark.slow
def test_hist_random_hyperparameter_search(hist_model) -> None:
    rf = hist_model
    rf.random_hyperparameter_search(3)
    assert(rf.model.get_params()["histgradientboostingregressor__max_iter"] != 100) #checking if default value no longer active

@pytest.mark.slow
def test_evolve_hyperparameter_search(load_database) -> None:
    rf = load_database
    rf.evolve_hyperparameter_search(3, 3)
    assert(rf.model.get_params()["randomforestregressor__n_estimators"] != 100) #checking if default value no longer active

@pytest.mark.slow
def test_save_true_vs_predicted(load_database) -> None:
    rf = load_database
    rf.random_hyperparameter_search(3)
    rf.save_true_vs_predicted("TruePred.png")
    assert(os.path.isfile("./train_TruePred.png"))
    assert(os.path.isfile("./test_TruePred.png"))

#test cal test data
def test_cal__init__(load_cal_database) -> None:
    """
    Test init for loading cal data
    """
    rf = load_cal_database
    assert(rf.model_filename == None or rf.model != None)

def test_cal_get_test_train_split(load_cal_database) -> None:
    rf = load_cal_database
    test_size = 0.5
    rf.get_test_train_split(test_size)
    assert(len(rf.X_test)>0)
    assert(len(rf.X_test)==len(rf.y_test))
    assert(len(rf.X_train)>0)
    assert(len(rf.X_train)==len(rf.y_train))
    lval = np.abs(test_size*len(rf.X_train) - (1-test_size)*len(rf.X_test))
    rval = 1e-3 * len(rf.X_test)
    assert(lval < rval)

@pytest.mark.slow
def test_cal_random_hyperparameter_search(load_cal_database) -> None:
    rf = load_cal_database
    rf.random_hyperparameter_search(3)
    assert(rf.model.get_params()["randomforestregressor__n_estimators"] != 100) #checking if default value no longer active

@pytest.mark.slow
def test_cal_evolve_hyperparameter_search(load_cal_database) -> None:
    rf = load_cal_database
    rf.evolve_hyperparameter_search(3, 3)
    assert(rf.model.get_params()["randomforestregressor__n_estimators"] != 100) #checking if default value no longer active

@pytest.mark.slow
def test_cal_hist_evolve_hyperparameter_search(hist_model) -> None:
    rf = hist_model
    rf.evolve_hyperparameter_search(3, 3)
    assert(rf.model.get_params()["histgradientboostingregressor__max_iter"] != 100) #checking if default value no longer active

@pytest.mark.slow
def test_cal_save_true_vs_predicted(load_cal_database) -> None:
    rf = load_cal_database
    rf.random_hyperparameter_search(3)
    rf.save_true_vs_predicted("TruePred.png")
    assert(os.path.isfile("data/train_TruePred.png"))
    assert(os.path.isfile("data/test_TruePred.png"))
