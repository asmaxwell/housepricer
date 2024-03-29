"""
Created by Andy S Maxwell 14/02/2024
Test flask app interface for housepricer
"""
from app import app
from housepricer import modeler

def test_parse_data() -> None:
    test_data = {"date" : "2021-01-05"
                 ,"postcode" :  "BS8 1TQ"
                 , "housetype" : "S", "newbuild" : "No"
                 , "estatetype" : "Leasehold", "transactiontype" : "B"}
    out = app.parse_data(test_data)
    assert out==[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2021, 1, 5, 358322, 173531]

def test_prediction_string() -> None:
    model = modeler.trainer("./data/", "./data/", "RandomModel.save", None, False, "notrandom")
    model.is_model_fitted()
    test_data = {"date" : "2005-01-05"
                 ,"postcode" :  "BS8 3PH"
                 , "housetype" : "D", "newbuild" : "No"
                 , "estatetype" : "Leasehold", "transactiontype" : "A"}
    
    model_pred_string = app.prediction_string(test_data, model)
    print(model_pred_string)
    assert model_pred_string[0:13]=="Prediction: £"

    val = float(model_pred_string[13:].replace(',',''))
    print(val)
    assert ( (val>1000) and (val<10000000) )