"""
Created by Andy S Maxwell 19/10/2023
Test file for house price trainer
"""
import pytest
import numpy as np
import pandas as pd
import housepricer.data_wrangling as hpt

### setup - load database
@pytest.fixture(scope='session') 
def load_database() -> None:
    wr = hpt.wrangling("data/")
    yield wr
    return

###test __init__
def test__init__(load_database) -> None:
    """
    This test also tests .load_data() and .load_model()
    """
    wr = load_database
    assert(len(wr.data)>0)
    assert(len(wr.features) >0)
    assert(len(wr.numerical_features) >0)
    assert(len(wr.categorical_features) >0)
    return


### def data_pruning(house_data : pd.DataFrame, feature_list : list[str]) -> None:
def test_data_pruning_features(load_database) -> None:
    """
    Tests if the correct features are in the DataFrame
    """
    wr = load_database
    #make test dataframe
    data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45],
    "weight": [5, 10, 15],
    "price_paid": [5, 15, 20],
    "deed_date" : ["01/10/2020", "07/10/2020", "14/10/2020"]
    }
    df = pd.DataFrame(data, index=["day1", "day2", "day3"])
    #select keys/ features
    keys = ["calories", "weight", "year", "month", "day"]
    #prune to these features
    wr.data = df
    wr.features = keys
    df_pruned, y_DF = wr.data_pruning()
    for feature, key in zip(df_pruned.keys(), keys):
        assert feature == key

def test_data_pruning_check_dates(load_database) -> None:
    """
    Test if the dates are converted correctly
    """
    wr = load_database
    #make test dataframe
    data = {
    "price_paid": [5, 15, 20],
    "deed_date" : ["05/10/2020", "07/01/2025", "14/11/2001"]
    }
    df = pd.DataFrame(data, index=["day1", "day2", "day3"])
    keys = ["year", "month", "day"]
    #prune to these features
    wr.data = df
    wr.features = keys
    df_pruned, y_DF = wr.data_pruning()
    
    #manually added data for year days month to check
    days = [5, 7, 14]
    months = [10, 1, 11] #incorrect format added at the end to see what happens
    years = [2020, 2025, 2001]

    for day, month, year, df_day, df_month, df_year in zip(days, months, years
        , df_pruned.loc[:,"day"], df_pruned.loc[:,"month"], df_pruned.loc[:,"year"]):
        assert day == df_day
        assert month == df_month
        assert year == df_year
def test_data_pruning_postcode(load_database) -> None:
    """
    Test postcodes converted correctly
    """
    wr = load_database
    #make test dataframe
    data = {
    "postcode": ["BS1 2AN", "BS8 1TQ", "BS8 3PH", "NR17 2LP"],
    "price_paid": [5, 15, 20, 25],
    "deed_date" : ["01/10/2020", "07/10/2020", "13/05/2023", "02/12/1991"]
    }
    df = pd.DataFrame(data)
    #prune to these features
    wr.data = df
    wr.features = ["latitude", "longitude"]
    df_pruned, y_DF = wr.data_pruning()
    df_pruned = df_pruned.sort_values(by = "latitude").reset_index(drop=True)
    print(df_pruned)
    assert df_pruned.loc[0, "latitude"] == 355944
    assert df_pruned.loc[1, "latitude"] == 358322
    assert df_pruned.loc[2, "latitude"] == 358963    
    assert df_pruned.loc[3, "latitude"] == 609588

    assert df_pruned.loc[0, "longitude"] == 172967
    assert df_pruned.loc[1, "longitude"] == 173531
    assert df_pruned.loc[2, "longitude"] == 173041    
    assert df_pruned.loc[3, "longitude"] == 295701


###def data_encoding(house_data: pd.DataFrame, categorical_features : list[str], numerical_features : list[str])  -> pd.DataFrame:
def test_data_encoding_categories(load_database) -> None:
    """
    check categories encoded in simple example
    """
    wr = load_database
    #make test dataframe
    data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45],
    "weight": [5, 10, 15],
    "type" : ["hard", "soft", "hard"]
    }
    df = pd.DataFrame(data, index=["day1", "day2", "day3"])
    wr.categorical_features = ["type"]
    wr.numerical_features = ["calories", "weight", "duration"]
    df_encoded = wr.data_encoding(df, 1)
    print(df_encoded)
    
    assert(df_encoded.loc["day1",0]==1.0)
    assert(df_encoded.loc["day2",0]==0)
    assert(df_encoded.loc["day3",0]==1.0)
    assert(df_encoded.loc["day1",1]==0)
    assert(df_encoded.loc["day2",1]==1.0)
    assert(df_encoded.loc["day3",1]==0.0)


### 2 - test def get_XY(self) -> None:
def test_get_XY(load_database) -> None:
    wr = load_database
    assert(len(wr.X)>0)
    assert(len(wr.X)==len(wr.y))

def test_get_test_train_split(load_database) -> None:
    wr = load_database
    test_size = 0.5
    wr.get_test_train_split(test_size)
    assert(len(wr.X_test)>0)
    assert(len(wr.X_test)==len(wr.y_test))
    assert(len(wr.X_train)>0)
    assert(len(wr.X_train)==len(wr.y_train))
    assert(np.abs(test_size*len(wr.X_train) - (1-test_size)*len(wr.X_test)) <1e-3 * len(wr.X_test))
