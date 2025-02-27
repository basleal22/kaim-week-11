import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
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
        