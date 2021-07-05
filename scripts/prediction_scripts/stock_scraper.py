import os
import requests
import json

import pandas as pd
import pandas_market_calendars as mcal

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import av_scraper

API_URL = "https://cloud.iexapis.com/"
with open('config.json') as config_file:
    data = json.load(config_file)
API_TOKEN = data['TOKEN']      # test token begins with T

_dir = os.getcwd()
dataset_dir = os.path.join(_dir, 'datasets')
scraped_dir = os.path.join(dataset_dir, 'scraped')


def fetch_data(symbol, dates):
    frames = {}
    for date in dates:
        URL_ENDPOINT = f"{API_URL}stable/stock/{symbol}/chart/date/{date}?token={API_TOKEN}"
        resp = requests.get(URL_ENDPOINT)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        df['time'] = df['date'] + " " + df['minute'] + ":00"
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        frames[date] = df

    total_df = pd.concat(frames.values()).set_index(keys='time')
    return frames, total_df


def save_data(dataframe):
    for key, value in dataframe.items():
        save_path = os.path.join(scraped_dir, key)
        file_name = os.path.join(save_path, f'{key}-total-data.csv')
        with open(file_name, 'a', newline='') as f:
            value.to_csv(f, header=(f.tell()==0))


def latest_data(symbol, last_30_days):
    last_day = last_30_days[-1]
    save_path = os.path.join(scraped_dir, symbol)
    file_name = os.path.join(save_path, f'{symbol}-total-data.csv')
    av_scrape_bool = False
    try:
        latest_date_entry = pd.to_datetime(pd.read_csv(file_name)['time'].iloc[-1]).strftime('%Y%m%d')
        try:
            dates_to_fetch = last_30_days[last_30_days.index(latest_date_entry) + 1:]
        except ValueError:      # if the last recorded date is not in the last 30 days
            dates_to_fetch = last_30_days
    except FileNotFoundError:
        dates_to_fetch = []     # if cannot find the file then there is NO data
        av_scrape_bool = True
    return dates_to_fetch, av_scrape_bool


def run(symbols):
    nyse = mcal.get_calendar('NYSE')
    last_30_days = nyse.schedule(start_date=datetime.now()-timedelta(days=30), end_date=datetime.now()).index.strftime('%Y%m%d').tolist()
    combined_dfs = {}
    for symbol in symbols:
        '''checks whether fetching data is necessary and if so which dates to fetch'''
        dates_to_fetch, av_scrape_bool = latest_data(symbol, last_30_days)
        if len(dates_to_fetch) > 0 and not av_scrape_bool:
            print(f"Fetching data for {symbol} between {dates_to_fetch[0]} - {dates_to_fetch[-1]} ...")
            '''get data and store in dictionary so it's easy to access each symbol and a corresponding day'''
            combined_dfs[symbol] = fetch_data(symbol, dates_to_fetch)[1]
            '''save data a csv'''
        elif av_scrape_bool:
            print(f"Fetching data for {symbol} through Alpha-Vantage's 2 year API...")
            av_scraper.run(months=24, symbols=[symbol])
        else:
            print(f"Data for {symbol} is up to date!")
    save_data(combined_dfs)


if __name__=='__main__':
	# 'AAPL', 'TSLA', 'GME', 'ABNB', 'PLTR', 'ETSY', 'ENPH', 'GOOG', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'
    symbols = [
        'AAPL', 'AMC'
    ]       # 13 tickers use ~20% of api usage
    run(symbols)

