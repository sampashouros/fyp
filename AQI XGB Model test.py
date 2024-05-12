# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:32:41 2024

@author: spash
"""
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df=pd.read_csv(r"C:\Users\spash\Documents\AQI DATA\AQI DATA.csv")
dfgrouped=df.groupby(df['Location'])
print(dfgrouped['Dominant Pollutant'].value_counts())
df.dropna(axis=0, inplace=True)


new_df = df.drop(['Latitude', 'Longitude', 'Category', 'Dominant Pollutant', 'DateTime', 'Unnamed: 0'], axis=1)


brooklyn_df = new_df[new_df['Location'] == 'Brooklyn']
manhattan_df = new_df[new_df['Location'] == 'Manhattan']
bronx_df = new_df[new_df['Location'] == 'Bronx']
queens_df = new_df[new_df['Location'] == 'Queens']
staten_island_df = new_df[new_df['Location'] == 'Staten Island']


pollutants = ["co (PARTS_PER_BILLION)", "no2 (PARTS_PER_BILLION)", "o3 (PARTS_PER_BILLION)", "pm10 (MICROGRAMS_PER_CUBIC_METER)", "pm25 (MICROGRAMS_PER_CUBIC_METER)", "so2 (PARTS_PER_BILLION)"]


location_dfs = {
    'Brooklyn': brooklyn_df,
    'Manhattan': manhattan_df,
    'The Bronx': bronx_df,
    'Queens': queens_df,
    'Staten Island': staten_island_df
}

lag_days = [1, 2, 3]
rolling_window = 9

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
import plotly.graph_objects as go

def plot_predictions_sorted(actual, predicted, location):
    #combine actual and predicted into a df
    combined_df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    
    # sort the df
    combined_df.sort_values(by='Actual', inplace=True)

    #reset index to get a proper line plot after sort
    combined_df.reset_index(drop=True, inplace=True)
    
    #create the traces for the sorted values
    trace1 = go.Scatter(x=combined_df.index, y=combined_df['Actual'], mode='lines+markers', name='Actual')
    trace2 = go.Scatter(x=combined_df.index, y=combined_df['Predicted'], mode='lines+markers', name='Predicted')
    
    #layout of the plot
    layout = go.Layout(
        title=f"{location} AQI Prediction vs Actual - Sorted",
        xaxis_title="Sorted Index",
        yaxis_title="AQI Value",
        legend_title="Legend",
        hovermode='closest'
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    fig.show(renderer="browser")



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
    # from xgboost import cv


    
    #intitialise and train the XGBoost regressor model
    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = 0.05, min_child_weight = 10,
                             max_depth = 3, n_estimators = 100, gamma = 10, alpha = 0.5)
    model.fit(X_train, y_train)
    # feature importance check
    # feature_importances = model.feature_importances_
    # features_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    # features_df = features_df.sort_values(by='Importance', ascending=False)
    # print(f"{location} - Feature Importances:\n{features_df}\n")
    
    #check for overfitting
    y_train_pred = pd.Series(model.predict(X_train), index=y_train.index)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    train_r2 = r2_score(y_train, y_train_pred)  # R^2 score for training set
    
    # prediction and evaluation
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)  # R^2 score
    
    print(f"{location} - Train RMSE: {train_rmse}")
    print(f"{location} - Train R^2 Score: {train_r2}")

    
    print(f"{location} - RMSE: {rmse}")
    print(f"{location} - R^2 Score: {r2}")
    
    overfit_metric = train_r2 - r2
    print(f"{location} - Overfitting Metric (ΔR^2): {overfit_metric}")
    
    plot_predictions_sorted(y_test, y_pred, location)
    
    #store updated data and model info
    location_dfs[location] = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'model': model, 'rmse': rmse
    }
