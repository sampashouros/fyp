#-*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:12:23 2024

@author: spash
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, cross_val_score

#load data
df = pd.read_csv(r"C:\Users\spash\Documents\AQI DATA\AQI DATA.csv")

#drop cols not needed
df.drop(['Latitude', 'Longitude', 'DateTime', 'Category', 'Dominant Pollutant'], axis=1, inplace=True)

#convert 'Location' to a numeric value
df['Location'], _ = pd.factorize(df['Location'])
pollutants = ["co (PARTS_PER_BILLION)", "no2 (PARTS_PER_BILLION)", "o3 (PARTS_PER_BILLION)", "pm10 (MICROGRAMS_PER_CUBIC_METER)", "pm25 (MICROGRAMS_PER_CUBIC_METER)", "so2 (PARTS_PER_BILLION)"]
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

#split the data into features and target
X = df.drop('AQI', axis=1)
y = df['AQI']

#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#define the bounds for bayesian optimisation
pbounds = {
    'learning_rate': (0.01, 1.0),
    'n_estimators': (100, 1000),
    'max_depth': (3, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'gamma': (0, 5)
}

#bayesian optimisation funct
def xgb_hyper_param(learning_rate, n_estimators, max_depth, subsample, colsample_bytree, gamma):
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)

    reg = XGBRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma
    )

    #cross-validation for model evaluation using negative mean squared error
    return np.mean(cross_val_score(reg, X_train, y_train, cv=3, scoring='neg_mean_squared_error'))

#initialize and run bayesian osptimisation
optimizer = BayesianOptimization(
    f=xgb_hyper_param,
    pbounds=pbounds,
    random_state=1,
)
optimizer.maximize(init_points=25, n_iter=75)

#best params found
print("Best parameters:", optimizer.max)