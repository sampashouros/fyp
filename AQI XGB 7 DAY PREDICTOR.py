# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:12:23 2024

@author: spash
"""
import pandas as pd
import numpy as np
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
    
    #split the data into features and target variable
    X = processed_df.drop('AQI', axis=1)
    y = processed_df['AQI']
    
    #split the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = 0.01,
                             max_depth = 10, n_estimators = 844, gamma = 3)
    model.fit(X_train, y_train)

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

def forecast_next_9_intervals(model, latest_data, pollutants, lag_days, rolling_window):
    interval_predictions = []
    
    for _ in range(9):
        #prep the features for the next interval's prediction
        next_interval_features = add_lag_and_rolling_avg(latest_data, pollutants, lag_days, rolling_window).iloc[-1:]
        
        #predict the next interval AQI
        next_interval_aqi = model.predict(next_interval_features.drop('AQI', axis=1, errors='ignore'))
        
        #store the interval prediction
        interval_predictions.append(next_interval_aqi.item())
        
        #sim updating the dataset with the new prediction for the next prediction
        new_row = {**next_interval_features.iloc[-1].to_dict(), 'AQI': next_interval_aqi.item()}
        new_row_df = pd.DataFrame([new_row], index=[latest_data.index[-1] + 1 if len(latest_data.index) > 0 else 0])
        latest_data = pd.concat([latest_data, new_row_df])
    
    #put interval predictions to a 3x3 matrix (3 days, 3 intervals per day)
    interval_predictions_matrix = np.array(interval_predictions).reshape(3, 3)
    
    #calculate daily AQI averages from the interval predictions
    daily_aqi_averages = np.mean(interval_predictions_matrix, axis=1)
    
    return daily_aqi_averages

location_coords = {
    'Brooklyn': '40.678177,-73.94416',
    'The Bronx': '40.840347,-73.876969',
    'Manhattan': '40.787534,-73.961126',
    'Staten Island': '40.599252,-74.11424',
    'Queens': '40.725119,-73.788628'
}

forecast_data = []

# The main loop for each location
for location, info in location_dfs.items():
    latest_data = info['X_test'].copy()
    latest_data['AQI'] = info['y_test']

    #9 intervals forecast (3 a day for 3 days)
    daily_aqi_averages = forecast_next_9_intervals(info['model'], latest_data, pollutants, lag_days, rolling_window)
    
    # Collecting forecast data for each day
    for day, avg_aqi in enumerate(daily_aqi_averages, 1):
        forecast_data.append({
            'Day': f"Day {day}",
            'Location': location,
            'Long-Lat': location_coords[location],
            'AQI Forecast': f"{avg_aqi:.2f}"
        })

#put into a dataframe
forecast_df = pd.DataFrame(forecast_data)

print(forecast_df)
forecast_df.to_csv(r"C:\Users\spash\Documents\AQI DATA\FORECAST DATA.csv")