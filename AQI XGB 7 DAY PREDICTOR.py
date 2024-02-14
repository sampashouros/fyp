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
    # Adding lag features and forward filling the null values
    for lag in lag_days:
        df[f'AQI_lag_{lag}'] = df['AQI'].shift(lag).fillna(method='ffill')
    # Adding a rolling average for AQI and filling nulls with the column mean
    df[f'AQI_rolling_avg_{rolling_window}d'] = df['AQI'].rolling(window=rolling_window).mean().fillna(df['AQI'].mean())
    
    for pollutant in pollutants:
        for lag in lag_days:
            df[f'{pollutant}_lag_{lag}'] = df[pollutant].shift(lag).fillna(method='ffill')
        # Adding rolling averages for each pollutant and filling nulls with the column mean
        df[f'{pollutant}_rolling_avg_{rolling_window}d'] = df[pollutant].rolling(window=rolling_window).mean().fillna(df[pollutant].mean())
    
    return df


# Preprocessing and feature engineering for each location
for location in location_dfs:
    # Apply feature engineering and drop the 'Location' column
    processed_df = add_lag_and_rolling_avg(location_dfs[location].copy(), pollutants, lag_days, rolling_window)
    processed_df.drop(['Location'], axis=1, inplace=True)
    
    # Splitting the data into features and target variable
    X = processed_df.drop('AQI', axis=1)
    y = processed_df['AQI']
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the XGBoost regressor model
    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                             max_depth = 5, alpha = 10, n_estimators = 100)
    model.fit(X_train, y_train)

    # Prediction and evaluation
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"{location} - RMSE: {rmse}")
    
    # Store updated data and model info
    location_dfs[location] = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'model': model, 'rmse': rmse
    }

def forecast_next_21_intervals(model, latest_data, pollutants, lag_days, rolling_window):
    interval_predictions = []
    
    for _ in range(21):
        # Prepare the features for the next interval's prediction
        next_interval_features = add_lag_and_rolling_avg(latest_data, pollutants, lag_days, rolling_window).iloc[-1:]
        
        # Predict the next interval's AQI
        next_interval_aqi = model.predict(next_interval_features.drop('AQI', axis=1, errors='ignore'))
        
        # Store the interval prediction
        interval_predictions.append(next_interval_aqi.item())
        
        # Simulate updating the dataset with the new prediction for the next prediction
        new_row = {**next_interval_features.iloc[-1].to_dict(), 'AQI': next_interval_aqi.item()}
        new_row_df = pd.DataFrame([new_row], index=[latest_data.index[-1] + 1 if len(latest_data.index) > 0 else 0])
        latest_data = pd.concat([latest_data, new_row_df])
    
    # Reshape interval predictions to a 7x3 matrix (7 days, 3 intervals per day)
    interval_predictions_matrix = np.array(interval_predictions).reshape(7, 3)
    
    # Calculate daily AQI averages from the interval predictions
    daily_aqi_averages = np.mean(interval_predictions_matrix, axis=1)
    
    return daily_aqi_averages


for location, info in location_dfs.items():
    print(f"Forecasting AQI for {location} over the next 21 intervals (7 days):")
    latest_data = info['X_test'].copy()  # Assuming X_test contains the most recent data; adjust as needed
    latest_data['AQI'] = info['y_test']  # This line might need adjustment based on how 'y_test' is structured
    
    # Make sure 'latest_data' includes all necessary columns for prediction,
    # especially if 'y_test' only contains the AQI values without other features.

    # Execute the 21 intervals forecast
    daily_aqi_averages = forecast_next_21_intervals(info['model'], latest_data, pollutants, lag_days, rolling_window)
    
    # Output the daily AQI averages for the next 7 days
    for day, avg_aqi in enumerate(daily_aqi_averages, 1):
        print(f"Day {day}: Average AQI = {avg_aqi:.2f}")

    print("\n")  # For better separation between locations
