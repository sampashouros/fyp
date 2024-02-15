# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:32:41 2024

@author: spash
"""
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"C:\Users\spash\Documents\AQI DATA\AQI DATA.csv")

df.dropna(axis=0, inplace=True)


new_df = df.drop(['Latitude', 'Longitude', 'Category', 'Dominant Pollutant', 'DateTime'], axis=1)


brooklyn_df = new_df[new_df['Location'] == 'Brooklyn']
manhattan_df = new_df[new_df['Location'] == 'Manhattan']
bronx_df = new_df[new_df['Location'] == 'Bronx']
queens_df = new_df[new_df['Location'] == 'Queens']
staten_island_df = new_df[new_df['Location'] == 'Staten Island']


pollutants = ["co (PARTS_PER_BILLION)", "no2 (PARTS_PER_BILLION)", "o3 (PARTS_PER_BILLION)", "pm10 (MICROGRAMS_PER_CUBIC_METER)", "pm25 (MICROGRAMS_PER_CUBIC_METER)", "so2 (PARTS_PER_BILLION)"]
lag_days = [1, 2, 3]
rolling_window = 7

location_dfs = {
    'Brooklyn': brooklyn_df,
    'Manhattan': manhattan_df,
    'The Bronx': bronx_df,
    'Queens': queens_df,
    'Staten Island': staten_island_df
}

def add_lag_and_rolling_avg(df, pollutants, lag_days, rolling_window):
    # add lag features and forward filling the null values
    for lag in lag_days:
        df[f'AQI_lag_{lag}'] = df['AQI'].shift(lag).fillna(method='ffill')
    # add a rolling average for AQI and filling nulls with the column mean
    df[f'AQI_rolling_avg_{rolling_window}d'] = df['AQI'].rolling(window=rolling_window).mean().fillna(df['AQI'].mean())
    
    for pollutant in pollutants:
        for lag in lag_days:
            df[f'{pollutant}_lag_{lag}'] = df[pollutant].shift(lag).fillna(method='ffill')
        # add rolling averages for each pollutant and filling nulls with the column mean
        df[f'{pollutant}_rolling_avg_{rolling_window}d'] = df[pollutant].rolling(window=rolling_window).mean().fillna(df[pollutant].mean())
    
    return df


#preprocessing and feature engineering for each location
for location in location_dfs:
    # feature engineering and drop the loc column
    processed_df = add_lag_and_rolling_avg(location_dfs[location].copy(), pollutants, lag_days, rolling_window)
    processed_df.drop(['Location'], axis=1, inplace=True)
    
    #splitthe data into features and target variable
    X = processed_df.drop('AQI', axis=1)
    y = processed_df['AQI']
    
    #split the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #intitialise and train the XGBoost regressor model
    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                             max_depth = 5, alpha = 10, n_estimators = 100)
    model.fit(X_train, y_train)
    # feature importance check
    feature_importances = model.feature_importances_
    features_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    features_df = features_df.sort_values(by='Importance', ascending=False)
    print(f"{location} - Feature Importances:\n{features_df}\n")
    #prediction and evaluation
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"{location} - RMSE: {rmse}")
    
    #store updated data and model info
    location_dfs[location] = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'model': model, 'rmse': rmse
    }
