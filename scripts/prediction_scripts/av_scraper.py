import math
import os
import csv
import time
import glob
from copy import deepcopy

from datetime import datetime
import dateutil.relativedelta

import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# globals
# Paths
_dir = os.getcwd()
dataset_dir = os.path.join(_dir, 'datasets')
scraped_dir = os.path.join(dataset_dir, 'scraped')

# Initialisations
av_api = 'QEI5J3EC7D5A94JZ'
ts = TimeSeries(key=av_api, output_format='csv', indexing_type='date')
cc = CryptoCurrencies(key=av_api, output_format='pandas')



def initialise(months):
	# Variables
	MONTHS = str(months)       # Data from 6 months ago to now

	DATES_INTERVALS = {}
	INTERVALS = [
		['1min'], ['1min'], ['5min'], ['5min'], ['5min'], ['15min'], ['15min'], ['15min'], ['15min'], ['30min'], ['30min'], ['30min'],
		['30min'], ['30min'], ['30min'], ['30min'], ['30min'], ['60min'], ['60min'], ['60min'], ['60min'], ['60min'], ['60min'], ['60min']
	]

	for num in range(24):
		num += 1
		dates = []
		for i in range(num):
			if i < 12:
				dates.append('year1month' + str(i + 1))
			else:
				dates.append('year2month' + str(i + 1 - 12))
		DATES_INTERVALS[str(num)] = [dates, INTERVALS[num - 1]]

	SLICE_DATES = DATES_INTERVALS[MONTHS][0]
	INTERVAL = DATES_INTERVALS[MONTHS][1]
	return SLICE_DATES, '1min'


def fetch_data(slice_dates, symbols, get_crypto, repeat_attempt=False):
    for symbol in symbols:
        print(f"Fetching {symbol} data for months: {slice_dates} at 1min intervals...")
        slice_dates_left = slice_dates.copy()
        for slices in slice_dates:
            if not check_exists(symbol=symbol, month=slice_dates.index(slices)):
                if not get_crypto:
                    fetch = ts.get_intraday_extended(symbol=symbol, slice=slices, interval='1min')
                    fetch_copy = deepcopy(list(fetch[0]))
                    try:
                        row = fetch_copy[1]
                        if len(row[0]) == 232:
                            print(row)
                            print("API Limit Reached")
                            if repeat_attempt:
                                time.sleep(60)
                                fetch_data(slice_dates_left, symbol, repeat_attempt=True)
                            else:
                                fetch_data(slice_dates_left, symbol, repeat_attempt=True)
                        else:
                            save_csv(fetch_copy, symbol=symbol, month=slice_dates.index(slices))
                            slice_dates_left.remove(slices)
                    except:
                        pass
                elif get_crypto:
                    fetch = cc.get_digital_currency_daily(symbol=symbol, market='CNY')
            else:
                print("Data is already stored...")


def make_dataframe(symbol, slice_dates):
    print(f"Making {len(slice_dates)} month {symbol} Dataframe...")
    ticker_dir = os.path.join(scraped_dir, symbol)
    all_files = glob.glob(ticker_dir + "\\*.csv")
    all_files.pop(-1)
    li = []
    for filename in all_files[-len(slice_dates):]:
        df = pd.read_csv(filename, header=0)
        df = df.reindex(index=df.index[::-1])    # Reversing our data as it gives us the end of the month and then works backwards
        li.append(df)
    frame = pd.concat(li, axis=0)
    frame.sort_values(by='time')
    frame.set_index(keys='time', inplace=True)

    new_path = os.path.join(ticker_dir, f'{symbol}-total-data.csv')
    frame.to_csv(new_path)


def check_exists(symbol, month):
    current_date = datetime.today()
    start_date = current_date - dateutil.relativedelta.relativedelta(months=month)
    ticker_dir = os.path.join(scraped_dir, symbol)
    path = os.path.join(ticker_dir, f'{start_date.strftime("%Y%m")}.csv')
    return os.path.isfile(path)


def save_csv(fetch, symbol, month):
    current_date = datetime.today()
    start_date = current_date - dateutil.relativedelta.relativedelta(months=month)
    ticker_dir = os.path.join(scraped_dir, symbol)
    path = os.path.join(ticker_dir, f'{start_date.strftime("%Y%m")}.csv')
    print(f"Saving fetched data to {path}")
    try:
        os.mkdir(ticker_dir)
    except FileExistsError:
        pass
    df = pd.DataFrame(fetch)
    df.to_csv(path, header=False, index=False)


def run(months, symbols=['AAPL']):
    slice_dates, interval = initialise(months)
    current_date = datetime.today()
    start_date = current_date - dateutil.relativedelta.relativedelta(months=len(slice_dates))
    print("Fetching Data...")
    get_crypto = True if symbols[0] == 'BTC' else False
    for symbol in symbols:
        try:
            os.remove(os.path.join(os.path.join(scraped_dir, symbol), f'{current_date.strftime("%Y%m")}.csv'))
        except:
            print("Latest Month Does Not Exist")
    fetch_data(slice_dates=slice_dates, symbols=symbols, get_crypto=get_crypto)
    print("Generating Dataframe...")
    for symbol in symbols:
        make_dataframe(symbol, slice_dates)


if __name__=='__main__':
	# 'AAPL', 'TSLA', 'GME', 'ABNB', 'PLTR', 'ETSY', 'ENPH', 'GOOG', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'
    symbols = [
        'AAPL', 'TSLA', 'GME', 'ABNB', 'PLTR', 'ETSY', 'ENPH', 'GOOG', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'
    ]
    coins = [
        'BTC', 'ETH', 'NANO', 'ADA', 'BAT',
        'ENJ', 'LINK', 'DOT', 'NMR', 'GRT'
    ]
    run(months=24, symbols=symbols)
    