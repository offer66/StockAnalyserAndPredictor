import os
import csv
from datetime import datetime as dt
import datetime

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError
import yfinance as yf

import matplotlib.pyplot as plt

_root = os.getcwd()

yf.pdr_override()


def get_data(ticker, timescale):
    today = dt.today()
    start_date = today - datetime.timedelta(days=365 * timescale)
    if "USD" in ticker:
        return pdr.DataReader(
            ticker,
            data_source="yahoo",
            start=start_date.strftime("%Y-%m-%d"),
        )["Adj Close"]
    else:
        return pdr.get_data_yahoo(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
        )["Adj Close"]


def calc_perc_change(data):
    perc_change = data.pct_change() * 100
    return perc_change


def get_days_col(data):
    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
    data["Weekday"] = data["Date"].dt.day_name()
    return data


def aggregate(data):
    return data[["Adj Close", "Weekday"]].groupby(["Weekday"]).mean()


def run(tickers, timescale):
    data = {}
    perc_change = {}
    weekday_perc_change = {}
    aggregated_data = {}
    for ticker in tickers:
        data[ticker] = get_data(ticker, timescale)
        perc_change[ticker] = calc_perc_change(data[ticker])
        weekday_perc_change[ticker] = get_days_col(perc_change[ticker])
        aggregated_data[ticker] = aggregate(weekday_perc_change[ticker])
    print(aggregated_data)


if __name__ == "__main__":
    tickers = ["GME", "BTC-USD", "ETH-USD"]
    run(tickers=tickers, timescale=0.25)
    plt.show()
