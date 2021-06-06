import datetime as dt
from datetime import datetime

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker 
from mpl_finance import candlestick_ohlc

from resistance_and_pivots import pivots_plot


def run(stocks, start_date):
    yf.pdr_override()
    smas_used = [10, 30, 50]
    start_date = start_date - dt.timedelta(days=max(smas_used))
    today = datetime.now()

    for stock in stocks:
        prices = pdr.get_data_yahoo(stock, start_date, today)

        fig, ax1 = plt.subplots() 

        for x in smas_used:
            sma = x
            prices[f'sma_{sma}'] = prices.iloc[:,4].rolling(window=sma).mean()
        
        # Bollinger Bands
        BB_period = 15      # moving average
        st_dev = 2
        prices[f'sma{BB_period}'] = prices.iloc[:,4].rolling(window=BB_period).mean()
        prices['stdev'] = prices.iloc[:,4].rolling(window=BB_period).std()
        prices['lower_band'] = prices[f"sma{BB_period}"] - (st_dev * prices['stdev'])
        prices['upper_band'] = prices[f"sma{BB_period}"] + (st_dev * prices['stdev'])  
        prices['Date'] = mdates.date2num(prices.index)

        period = 10
        k = 4
        d = 4

        prices['rol_high'] = prices['High'].rolling(window=period).max()
        prices['rol_low'] = prices['Low'].rolling(window=period).min()
        prices['stok'] = ((prices['Adj Close'] - prices['rol_low']) / (prices['rol_high'] - prices['rol_low'])) * 100
        prices['k'] = prices['stok'].rolling(window=k).mean()
        prices['d'] = prices['k'].rolling(window=d).mean()
        prices['gd'] = prices['High']   # stores green dots

        ohlc = []

        prices = prices.iloc[max(smas_used):]

        green_dot_date = []
        green_dot = []
        last_k = 0
        last_d = 0
        last_low = 0
        last_close = 0
        last_low_BB = 0

        for i in prices.index:
            append_me = prices['Date'][i], prices['Open'][i], prices['High'][i], prices['Low'][i], prices['Adj Close'][i], prices['Volume'][i],
            ohlc.append(append_me)

            # checks for green dot
            if prices['k'][i] > prices['d'][i] and last_k < last_d and last_k < 60:
                plt.plot(prices['Date'][i], prices['High'][i] + 1, marker='o', ms=4, ls='', color='g')

                green_dot_date.append(i)
                green_dot.append(prices['High'][i])
            
            # checks for lower Bollinger Band bounce
            if (
                (last_low < last_low_BB) or (prices['Low'][i] < prices['lower_band'][i])) and (prices['Adj Close'][i] > last_close and prices['Adj Close'][i] > prices['lower_band'][i]
            ):
                plt.plot(prices['Date'][i], prices['Low'][i] - 1, marker='o', ms=4, ls='', color='b')

            last_k = prices['k'][i]
            last_d = prices['d'][i]
            last_low = prices['Low'][i]
            last_close = prices['Adj Close'][i]
            last_low_BB = prices['lower_band'][i]

        
        # plot moving avgs
        for x in smas_used:
            sma = x
            prices[f'sma_{sma}'].plot(label='Close')
        prices['upper_band'].plot(label='close', color='lightgray')
        prices['lower_band'].plot(label='close', color='lightgray')
        
        # plot candles
        candlestick_ohlc(ax1, ohlc, width=.5, colorup='k', colordown='r', alpha=0.7)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(8))
        plt.tick_params(axis='x', rotation=45)


        # pivot points
        pivots_plot(df=prices, stock=stock, days=30)

if __name__=='__main__':
    stocks = ['AAPL']
    start_date = datetime(2020, 12, 1)
    run(stocks, start_date)
    plt.show()






