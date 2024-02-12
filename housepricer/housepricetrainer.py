"""
Created by Andy S Maxwell 19/10/2023
Script to train model on house prices in Bristol
"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
#import pgeocode as pgc
from housepricer import better_postcodes as bpc
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay
from sklearn import preprocessing



def data_pruning(house_data : pd.DataFrame, features : list[str]) -> pd.DataFrame:
    """
    Remove unnecessary features from data
    house_data: data (by view) to operate on directly
    features: the features that will be kept (ideally this will include the new features year, day and month)
    """
    if "latitude" in features or "longitude" in features:
        print("converting postcodes")
        #convert postcode to latitude and longitude
        # nomi = pgc.Nominatim('GB') #set country
        # #get list of postcodes only
        # postcode_list = house_data.loc[:,'postcode'].values.tolist()
        # geocode_data = nomi.query_postal_code(postcode_list)

        # #add lats and longs
        # house_data.loc[:,'latitude'] = geocode_data.loc[:,'latitude']
        # house_data.loc[:,'longitude'] = geocode_data.loc[:,'longitude']
        # pd.set_option("display.precision", 14)
        #print(house_data)
        converter = bpc.better_postcodes("../data/codepo_gb/Data/CSV/")
        postcode_list = house_data.loc[:,'postcode'].values.tolist()
        df_coords = converter.query_postcodes(postcode_list)
        house_data.loc[:,'latitude'] = df_coords.loc[:,'Eastings'] #similar to lat and long (Eastings and Northings)
        house_data.loc[:,'longitude'] = df_coords.loc[:,'Northings']


        #remove missing postcodes
        house_data = house_data[house_data.loc[:,'longitude'].notna()]

    if "day" in features or "month" in features or "year" in features:
        # get dates data
        house_data_dates = pd.DatetimeIndex(house_data.loc[:,'deed_date'], dayfirst=True)
        
        #split dates into three features year, month, and day, as more useful like this
        year_column = house_data_dates.year
        month_column = house_data_dates.month
        week_day_column = house_data_dates.day

        #add features
        house_data.loc[:,'year']=year_column
        house_data.loc[:,'month']=month_column
        house_data.loc[:,'day']=week_day_column

    #prune down to selected features
    house_data_pruned = house_data[features]
    return house_data_pruned, house_data['price_paid']

def data_encoding(house_data: pd.DataFrame, categorical_features : list[str], numerical_features : list[str], min_freq : int = 15)  -> pd.DataFrame:
    """
    Encode the categorical features of the data
    house_data : input panda data frame
    categorical_features : list of strings for categorical features
    numerical_features : list of strings for numerical features
    """
    X_numerical = house_data.loc[:,numerical_features]
    X_categorical = house_data.loc[:,categorical_features]
    row_names = X_categorical.index
    #encoding
    drop_enc = preprocessing.OneHotEncoder(min_frequency=1).fit(X_categorical.values.tolist())#preprocessing.OneHotEncoder(drop='first',handle_unknown='error',min_frequency=1).fit(X_categorical.values.tolist())
    X_categorical_transformed = drop_enc.transform(X_categorical.values.tolist()).toarray()
    X_categorical = pd.DataFrame(X_categorical_transformed, index=row_names)
    
    #merge back together and return
    return pd.concat([X_categorical, X_numerical], axis="columns")


def plot_cross_validated_pred(y:list, y_pred:list) -> None:
    """
    Function to use matlib plot to show predicted vs true y and the residuals
    """
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        random_state=0,
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    plt.show()