"""
Created by Andy S Maxwell 19/10/2023
Class to deal with data on house prices in Bristol for training ensemble decision tree models
"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.datasets import fetch_california_housing
from . import better_postcodes as bpc
from sklearn import preprocessing
from sklearn import model_selection

class wrangling:
    load_cal_data : bool
    data_directory : str
    postcode_directory : str
    data : pd.DataFrame
    features : list[str]
    numerical_features : list[str]
    categorical_features : list[str]
    X : list[float]
    y : list[float]

    def __init__(self, data_directory: str
                 , postcode_directory : str = None
                 , load_cal_data : bool = False
                 )  ->None:
        """
            Init function, loading data, and postcodes. Test californian data can be loaded if specified
        """
        self.load_cal_data = load_cal_data
        self.data_directory = None if load_cal_data else data_directory
        self.postcode_directory = postcode_directory
        self.load_data()

        if self.load_cal_data:
            #cal features
            self.features = self.data[0].index
        else:
            # select and engineer features
            self.features = ['year', 'month', 'day', 'latitude', 'longitude'
                ,'property_type','new_build', 'estate_type'
                , 'transaction_category']

            # use dummy encoding for categories
            self.numerical_features = ['year', 'month', 'day', 'latitude', 'longitude']
            self.categorical_features = ['property_type', 'new_build','estate_type','transaction_category']

        self.get_XY()
        self.get_test_train_split(0.1)

        return
    def load_data(self) -> None:
        """
        Function to load house price data as member variable
        """
        if self.load_cal_data:
            X, y = fetch_california_housing(return_X_y=True, as_frame=True)
            data = [X, y]
        else:
            data = pd.read_csv(self.data_directory + "ppd_data.csv")
            # remove NaN and scale price
            data = data[data.loc[:,'postcode'].notna()]

            # scale price paid in millions
            #convert to float from int to suppress pandas error message
            data['price_paid'] = data['price_paid'].astype(float)
            data.loc[:,'price_paid'] = data.loc[:,'price_paid']/1.0e6

        self.data = data
        return
    
    def get_XY(self) -> None:
        """
        Do preprocessing, encoding and produce list for X and Y from Data
        """
        if self.load_cal_data:
            self.X = self.data[0]#.values.tolist()
            self.y = self.data[1]#.values.tolist()
        else:
            # split features into categorical and numerical
            X_dataframe, y_dataframe = self.data_pruning() if self.postcode_directory == None else self.data_pruning(self.postcode_directory) 
            
            # keep subset of data for now
            recent_years = X_dataframe['year']>1989 #data starts from around 1990.
            X_dataframe = X_dataframe[recent_years]

            self.y = y_dataframe.loc[recent_years].values.tolist()
            #encode data
            X_dataframe = self.data_encoding(X_dataframe)
            self.X = X_dataframe.values.tolist()
        return
    def get_test_train_split(self, test_size: float = 0.1) -> tuple[list]:
        """
        split data into test and train data sets
        """
        if len(self.X)>0 and len(self.y)>0:
            #test/train split
            return model_selection.train_test_split(self.X, self.y, test_size=test_size)
        else:
            print("Error X and Y not filled yet!")
            return () #returns empty tuple

    def data_pruning(self, postcode_directory: str = "data/codepo_gb/Data/CSV/") -> pd.DataFrame:
        """
        Remove unnecessary features from data
        house_data: data (by view) to operate on directly
        features: the features that will be kept (ideally this will include the new features year, day and month)
        """
        if "latitude" in self.features or "longitude" in self.features:
            print("converting postcodes")
            converter = bpc.better_postcodes(postcode_directory)
            postcode_list = self.data.loc[:,'postcode'].values.tolist()
            df_coords = converter.query_postcodes(postcode_list)
            self.data.loc[:,'latitude'] = df_coords.loc[:,'Eastings'] #similar to lat and long (Eastings and Northings)
            self.data.loc[:,'longitude'] = df_coords.loc[:,'Northings']


            #remove missing postcodes
            self.data = self.data[self.data.loc[:,'longitude'].notna()]

        if "day" in self.features or "month" in self.features or "year" in self.features:
            # get dates data
            house_data_dates = pd.DatetimeIndex(self.data.loc[:,'deed_date'], dayfirst=True)
            
            #split dates into three features year, month, and day, as more useful like this
            year_column = house_data_dates.year
            month_column = house_data_dates.month
            week_day_column = house_data_dates.day

            #add features
            self.data.loc[:,'year']=year_column
            self.data.loc[:,'month']=month_column
            self.data.loc[:,'day']=week_day_column

        #prune down to selected features
        house_data_pruned = self.data[self.features]
        return house_data_pruned, self.data['price_paid']

    def data_encoding(self, X_dataframe : pd.DataFrame, min_freq : int = 15)  -> pd.DataFrame:
        """
        Encode the categorical features of the data
        X_dataframe : panda data frame output of processing self.data with self.data_pruning()
        categorical_features : list of strings for categorical features
        numerical_features : list of strings for numerical features
        """
        X_numerical = X_dataframe.loc[:,self.numerical_features]
        X_categorical = X_dataframe.loc[:,self.categorical_features]
        row_names = X_categorical.index
        #encoding
        drop_enc = preprocessing.OneHotEncoder(min_frequency=min_freq).fit(X_categorical.values.tolist())#preprocessing.OneHotEncoder(drop='first',handle_unknown='error',min_frequency=1).fit(X_categorical.values.tolist())
        X_categorical_transformed = drop_enc.transform(X_categorical.values.tolist()).toarray()
        X_categorical = pd.DataFrame(X_categorical_transformed, index=row_names)
        
        #merge back together and return
        return pd.concat([X_categorical, X_numerical], axis="columns")

