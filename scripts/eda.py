import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
def extract_data(ticker,starttime,endtime):
    data = yf.download(ticker,start=starttime,end=endtime)
    return data
def closing_vis(data):
    closing_price = data['Close']
    tickers = ['TSLA','BND','SPY']
    #ploting closing price
    for ticker in tickers:
        plt.plot(closing_price.index,closing_price[ticker],label=ticker) 
    plt.title('closing price plot for all assests')
    plt.xlabel('Date')
    plt.ylabel('price(USD)')
    plt.legend()
    plt.grid()
    plt.show()
def percentage_change(data):
    closing_price = data['Close']
    tickers=['TSLA','BND','SPY']
    #checking precentage change
    percent_chg=closing_price.pct_change()*100
    #plot the daily return
    plt.figure(figsize=(10,6))
    for ticker in tickers:
        plt.plot(percent_chg.index,percent_chg[ticker], label=f"daily % change of{ticker}", alpha = 0.7)

    plt.title("tesla daily percentage change")
    plt.xlabel('Date')
    plt.ylabel('daily return(%)')
    plt.legend()
    plt.grid()
    plt.show()
def rolling_windows(data):
    tickers=['TSLA','BND','SPY']
    closing_price=data['Close']
    percent_chg=closing_price.pct_change()*100
    rolling_window=30
    rolling_mean = percent_chg.rolling(window=rolling_window).mean()
    rolling_std = percent_chg.rolling(window=rolling_window).std() 
    #plot the rolling std
    plt.figure(figsize=(12,6))
    for ticker in tickers:
        plt.plot(rolling_std.index,rolling_std[ticker],label=f'{ticker} {rolling_window} -days volatility',alpha=0.7)
    plt.title(f'{rolling_window} - Day rolling volatility of TSLA, SPY, and BND ')
    plt.xlabel('date')
    plt.ylabel('volatility %')
    plt.grid()
    plt.legend()
    plt.show()
def outliers_identify(data):
    tickers = ['TSLA', 'BND', 'SPY']
    closing_price = data['Close']
    
    # Calculate daily returns
    percent_chg = closing_price.pct_change() * 100  # Convert to percentage
    
    rolling_window = 30
    epsilon = 1e-8  # Small value to prevent division by zero

    # Compute rolling mean and standard deviation
    rolling_mean = percent_chg.rolling(window=rolling_window, min_periods=1).mean()
    rolling_std = percent_chg.rolling(window=rolling_window, min_periods=1).std() + epsilon  # Avoid division by zero

    # Define Z-score threshold (try lowering it)
    z_score_threshold = 2.5  

    # Compute Z-score (handle NaNs)
    z_scores = (percent_chg - rolling_mean) / rolling_std
    z_scores = z_scores.dropna()  # Drop NaNs to avoid errors

    # Identify outliers where abs(z-score) > threshold
    outliers = percent_chg[np.abs(z_scores) > z_score_threshold]

    # Drop rows where all tickers are NaN (no outliers)
    outliers = outliers.dropna(how='all')
    outliers_filled = outliers.fillna(0)  

    # Display results
    if outliers.empty:
        print("No significant outliers detected with Z-score threshold:", z_score_threshold)
    else:
        print("Outlier Days with Unusually High or Low Returns:")
        print(outliers_filled)
    #visualize the outliers
    plt.figure(figsize=(12,6))
    sns.boxplot(data=percent_chg)
    plt.title('Boxplot of Daily Returns')
    plt.xlabel('date')
    plt.ylabel('outliers')
    plt.grid()
    plt.legend()
    plt.show()
def seasonal_decomposition(data):
    tickers= ['TSLA', 'BND', 'SPY']
    plt.figure
    for ticker in tickers:
        closing=data['Close'][ticker].dropna()
        #perform seasonal decomposition
        decomposition = seasonal_decompose(closing,model='additive',period=30)
        decomposition.plot()
        plt.show()