"""
Created by Andy S Maxwell 11/02/2024
Class to convert postcode to latitude and longitude
"""
import pandas as pd
#import numpy as np
import os

class better_postcodes:
    
    def __init__(self, postcode_directory: str):
        self._postcode_directory = postcode_directory
        dataframe_list =[]
        for file in os.listdir(postcode_directory):
            filename = os.fsdecode(file)
            file_dataframe = pd.read_csv(postcode_directory+filename,header=None).loc[:,[0, 2, 3]]
            dataframe_list.append(file_dataframe)

        self.postcode_data = pd.concat(dataframe_list, axis=0, ignore_index=True)
        
    
    def query_postcodes(self, postcode) -> pd.DataFrame:
        if type(postcode) == str or type(postcode) == list:
            if type(postcode) == str:
                postcode = [postcode]
            index = self.postcode_data.loc[:,0].isin(postcode)
            data ={"Eastings" : self.postcode_data.loc[index,2].values.tolist(),
                "Northings": self.postcode_data.loc[index,3].values.tolist()}
            return pd.DataFrame( data )
        else:
            print("Warning incorrect postcode type given to query_postcodes")
            print("It should be a string or list of strings")
            data = {"Eastings" : [-1], 
                    "Northings" : [-1]}
            return pd.DataFrame(data)



    