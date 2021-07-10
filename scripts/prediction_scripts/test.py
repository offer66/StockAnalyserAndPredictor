import os
import time
import sys
sys.path.append("scripts\\analysis_scripts\\")
print(sys.version)
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import pandas_market_calendars as mcal
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

import model_comparison as validation_csv
import rsi as rsi_calc

# import initialisations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
yf.pdr_override()

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# some base directories
_root = os.getcwd()
_data = os.path.join(_root, 'datasets\\scraped\\')
_models = os.path.join(_root, 'models')

def get_yahoo_data():
    data_dict = {ticker: [] for ticker in symbols}
    for ticker in data_dict:
        data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
        data = data.reset_index()
        data = data.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj close', 'Volume': 'volume'})
        data = data.set_index('time')
        data_dict[ticker] = data
    # print(data_dict)
    return data_dict

def get_yahoo_data_2():
    data = pdr.get_data_yahoo(symbols, start=start_date, end=today)
    data = data.swaplevel(0, 1, axis=1)
    data = data.reset_index()
    data = data.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj close', 'Volume': 'volume'})
    data = data.set_index('time')
    return data

if __name__=="__main__":
    symbols = [
		'AAPL', 'GME', 'ABNB', 'PLTR', 'ETSY', 'ENPH', 'GOOG', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'
	]
    # symbols = [
	# 	'AAPL', 'TSLA'
    # ]

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=6 * 30)		# how far back we access yahoo finance stock EOD data
    
    # start = time.time()
    # data = get_yahoo_data()
    # print("Method 1: ", time.time() - start)
    # print(data)

    start = time.time()
    data = get_yahoo_data_2()
    print("Method 2: ", time.time() - start)
    print(data)