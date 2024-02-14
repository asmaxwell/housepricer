"""
Created by Andy S Maxwell 11/02/2024
Test for model class to train random forest model
"""
import pytest
import numpy as np
from housepricer import modeler


### setup - load database
@pytest.fixture(scope='session') 
def load_database() -> None:
    rf = modeler.random_forest("data/", "./")
    yield rf
    return
### 1 - test __init__
def test__init__(load_database) -> None:
    """
    This test also tests .load_data() and .load_model()
    """
    rf = load_database
    assert(len(rf.data)>0)
    assert(rf.model_filename == None or rf.model != None)
    assert(len(rf.features) >0)
    assert(len(rf.numerical_features) >0)
    assert(len(rf.categorical_features) >0)

### 2 - test def get_XY(self) -> None:
def test_get_XY(load_database) -> None:
    rf = load_database
    assert(len(rf.X)>0)
    assert(len(rf.X)==len(rf.y))

def test_get_test_train_split(load_database) -> None:
    rf = load_database
    test_size = 0.5
    rf.get_test_train_split(test_size)
    assert(len(rf.X_test)>0)
    assert(len(rf.X_test)==len(rf.y_test))
    assert(len(rf.X_train)>0)
    assert(len(rf.X_train)==len(rf.y_train))
    assert(np.abs(test_size*len(rf.X_train) - (1-test_size)*len(rf.X_test)) <1e-3 * len(rf.X_test))

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

@pytest.mark.slow
def test_random_hyperparameter_search(load_database) -> None:
    rf = load_database
    rf.random_hyperparameter_search(3)
    assert(rf.model.get_params()["randomforestregressor__n_estimators"] != 100) #checking if default value no longer active

@pytest.mark.slow
def test_evolve_hyperparameter_search(load_database) -> None:
    rf = load_database
    rf.evolve_hyperparameter_search(3, 3)
    assert(rf.model.get_params()["randomforestregressor__n_estimators"] != 100) #checking if default value no longer active