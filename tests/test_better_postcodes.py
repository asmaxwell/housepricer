"""
Created by Andy S Maxwell 11/02/2024
Test for class to convert postcode to latitude and longitude
"""

from housepricer import better_postcodes as bpc

### 1 - 

def test_query_postcodes() -> None:
    #test postcode conversion
    postcode = "BS1 2AN"
    
    converter = bpc.better_postcodes("../data/codepo_gb/Data/CSV/")

    df = converter.query_postcodes(postcode)
    assert df.loc[0, "Eastings"] == 358963
    assert df.loc[0, "Northings"] == 173041

    postcode_list = ["NR17 2LP", "BS8 3PH", "BS8 1TQ"]
    df = converter.query_postcodes(postcode_list)
    assert df.loc[0, "Eastings"] == 358322
    assert df.loc[0, "Northings"] == 173531
    assert df.loc[1, "Eastings"] == 355944
    assert df.loc[1, "Northings"] == 172967 
    assert df.loc[2, "Eastings"] == 609588
    assert df.loc[2, "Northings"] == 295701
