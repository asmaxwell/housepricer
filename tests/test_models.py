"""
Created by Andy S Maxwell 11/02/2024
Test for model class to train random forest model
"""

from housepricer import models

### 1 - test __init__
def test__init__() -> None:
    rf = models.random_forest("../data/", "./")
    assert(len(rf.data)>0)
    assert(rf.model_filename == None or self.model != None)
    assert(len(rf.features) >0)
    assert(len(rf.numerical_features) >0)
    assert(len(rf.categorical_features) >0)

### 2 - test def get_XY(self) -> None:
def test_get_XY() -> None:
