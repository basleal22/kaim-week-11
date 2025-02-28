import pandas as pd
import numpy as np
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error, mean_absolute_percentage_error
from math import sqrt
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
#we split our data into test and train
def split_test_train(data):
    split_ratio=0.8
    split_index = int(len(data)*split_ratio)
    #train_test
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]
    return train,test
def arima_model(train,test):
    tickers_close=['Close_BND', 'Close_SPY', 'Close_TSLA']
    for ticker in tickers_close:
        train_ticker=train[ticker].squeeze()
        arima_m = ARIMA(train_ticker,order=(1,1,1))
        arima_fit = arima_m.fit()
    return arima_fit
def sarima_model(train,test):
    tickers_close=['Close_BND', 'Close_SPY', 'Close_TSLA']
    for ticker in tickers_close:
        train_ticker=train[ticker].squeeze()
        sarima_m= SARIMAX(train_ticker,seasonal_order=(1,1,1,12))
        sarima_fit=sarima_m.fit()
    return sarima_fit
