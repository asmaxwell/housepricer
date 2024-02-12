"""
Created by Andy S Maxwell 11/02/2024
Test for model class to train random forest model
"""

from models import random_forest

### 1 - test __init__
def test___init__() -> None:
    rf = random_forest("../Data/", "./")
