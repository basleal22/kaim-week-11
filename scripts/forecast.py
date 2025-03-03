import pandas as pd
import numpy as np
import tensorflow as tf
#from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error, mean_absolute_percentage_error
from math import sqrt
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
#we split our data into test and train
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
def split_test_train(data):
    split_ratio=0.8
    split_index = int(len(data)*split_ratio)
    #train_test
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]
    return train,test
def arima_model(train,test):
    tickers_close=['Close_BND', 'Close_SPY', 'Close_TSLA']
    models={}
    for ticker in tickers_close:
        train_ticker=train[ticker].squeeze()
        arima_m = ARIMA(train_ticker,order=(0,1,2))
        arima_fit = arima_m.fit()
        models[ticker]=arima_fit
    return models
def sarima_model(train,test):
    tickers_close=['Close_BND', 'Close_SPY', 'Close_TSLA']
    models={}
    for ticker in tickers_close:
        train_ticker=train[ticker].squeeze()
        sarima_m= SARIMAX(train_ticker,order = (1,1,1),seasonal_order=(1,1,1,12))
        sarima_fit=sarima_m.fit()
        models[ticker]=sarima_fit
    return models
def opt_sarima_model(train,test):
    tickers_close=['Close_BND', 'Close_SPY', 'Close_TSLA']
    models={}
    for ticker in tickers_close:
        train_ticker=train[ticker].squeeze()
        sarima_m= SARIMAX(train_ticker,order = (1,1,1), seasonal_order=(0,1,2,12))
        sarima_fit=sarima_m.fit()
        models[ticker]=sarima_fit
    return models
#def auto_arima_model(train):
    """
    Fits auto-ARIMA models to the closing prices of specified tickers.

    Args:
        train (pandas.DataFrame): DataFrame containing the closing prices, 
                                    with columns 'Close_BND', 'Close_SPY', 'Close_TSLA'.

    Returns:
        dict: A dictionary where keys are ticker names and values are the fitted ARIMA models.
    """
    tickers_close = ['Close_BND', 'Close_SPY', 'Close_TSLA']
    best_models = {}

    for ticker in tickers_close:
        try: # Added try block to handle potential errors
            train_ticker = train[ticker].squeeze()

            # Auto-ARIMA model selection
            model = auto_arima(train_ticker,
                               start_p=0, max_p=5,
                               start_q=0, max_q=5,
                               d=None,  # Let auto_arima determine 'd'
                               seasonal=False,
                               stepwise=True, trace=True,
                               error_action="ignore",  # Add error handling
                               suppress_warnings=True) # Suppress harmless warnings

            best_models[ticker] = model
            print(f'Best ARIMA model for {ticker}: {model.order}')

        except Exception as e: # Catch any exceptions
            print(f"Error fitting ARIMA for {ticker}: {e}")
            best_models[ticker] = None # Store None in case of failure.

    return best_models
import itertools
def parameters(train,test):
    p_values = range(0, 3)  # Seasonal AR terms
    d_values = range(0, 2)  # Seasonal differencing
    q_values = range(0, 3)  # Seasonal MA terms
    s_values = [6, 12]  # Test different seasonality periods

    best_aic = float("inf")
    best_params = None

    for p, d, q, s in itertools.product(p_values, d_values, q_values, s_values):
        try:
            model = SARIMAX(train["Close_BND"], seasonal_order=(p, d, q, s))
            result = model.fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_params = (p, d, q, s)
        except:
            continue

    print(f"Best Seasonal Order: {best_params} with AIC={best_aic}")
def lstm_model(X_train,y_train,X_test,y_test):
    scaler=MinMaxScaler(feature_range=(0,1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  
    X_test_scaled = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))  
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1))
    

    model=Sequential([
        LSTM(50,activation = 'relu',return_sequences=True,input_shape=(1,X_train_scaled.shape[2])),
        Dropout(0.2),
        LSTM(50,return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)     
    ])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_train_scaled, y_train_scaled,epochs=20,batch_size=32,validation_data=(X_test_scaled,y_test_scaled))
    return model
