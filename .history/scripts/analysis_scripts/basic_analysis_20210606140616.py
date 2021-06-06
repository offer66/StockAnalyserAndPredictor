import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError
import yfinance as yf

import matplotlib.pyplot as plt


# Get portfolio data
_root = os.getcwd()
_portfolio = os.path.join(_root, 'datasets\\portfolio\\')
_plots = os.path.join(_portfolio, 'plots')

portfolio_data_raw = pd.read_csv(_portfolio + '\\portfolio_simple.csv')
portfolio_data = portfolio_data_raw.copy()


# Split portfolio data into useable data
symbols = list(portfolio_data['symbol'])
portfolio_data = portfolio_data.set_index('symbol')
portfolio_data['invested'] = portfolio_data['amount'] * portfolio_data['avg_price']
portfolio_weights = list(portfolio_data['invested'] / portfolio_data['invested'].sum())
portfolio_data['percentage'] = portfolio_weights
portfolio_weights = np.array([float('{:.4f}'.format(i)) for i in portfolio_weights])

plt.pie(portfolio_data['invested'], labels=portfolio_data.index, explode=[0.05 for i in range(len(portfolio_data['invested']))], autopct='%.2f%%', shadow=True, startangle=90)
plt.savefig(os.path.join(_plots, 'Portfolio_Stock_Distribution.png'))
print(symbols)


# Get sector data
sector_data = []
market_cap_data = []
for ticker in symbols:
    obj = yf.Ticker(ticker)
    try:
        sector_data.append(obj.info['sector'])
        market_cap_data.append(obj.info['marketCap'])
    except KeyError:
        sector_data.append('Crypto')
        market_cap_data.append(0)       # for now before i connect a crypto api

portfolio_data['sector'] = sector_data
portfolio_data['cap'] = market_cap_data
print(portfolio_data)

sector_count = {}
sector_set = set(sector_data)
for sector in sector_set:
    count = sector_data.count(sector)
    sector_count[sector] = count

sector_df = pd.DataFrame.from_dict(sector_count, orient='index')
print(sector_df)
# plt.pie(sector_df[0], labels=sector_df.index, explode=[0.1 for i in range(len(sector_df))], autopct='%.2f%%', shadow=True, startangle=90)
# plt.savefig(os.path.join(_plots, 'Portfolio_Sector_Distribution.png'))

portfolio_weights = np.array(portfolio_weights)
start_date = '2020-12-01'
today = datetime.today().strftime('%Y-%m-%d')
len_portfolio = len(symbols)
print(f"Have {len_portfolio} assets")


# Get prices
def get_prices(stocks, start, end, col='Adj Close'):
    try:
        data = pdr.get_data_yahoo(stocks, start, end)
    except RemoteDataError:
        data = pdr.DataReader(f'{stocks}-USD','yahoo', start, end)
    # data = web.DataReader(stocks, data_source='yahoo', start=start, end=end)
    data = data[col]
    print(data.head())
    print(data.tail())
    return data


# Visualise Data
def show_data(data, col):
    title = 'Portfolio ' + col + ' Price History'
    plt.figure(figsize=(12, 4))
    for stock in data.columns.values:
        plt.plot(data[stock], label=stock)
    
    plt.title(title)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(col + ' Price ($)', fontsize=18)
    plt.legend(data.columns.values, loc='upper left')


data = get_prices(stocks=symbols, start=start_date, end=today, col='Adj Close')
show_data(data=data, col='Adj Close')

# Calculate simple returns
daily_simple_returns = data.pct_change(1)
daily_simple_returns
# Correlation between stocks
daily_simple_returns.corr()
# Show Variance
daily_simple_returns.var()


# Stock Volatility (std)
daily_simple_returns.std()
most_volatile = max(daily_simple_returns.std())
least_volatile = min(daily_simple_returns.std())


# Visualise DSR
plt.figure(figsize=(12, 4))

for stock in daily_simple_returns.columns.values:
    plt.plot(daily_simple_returns[stock], lw=2, label=stock)

plt.legend(loc='upper right', fontsize=10)
plt.title('Volatility')
plt.xlabel('Date')
plt.ylabel('Daily Simple Returns')
plt.savefig(os.path.join(_plots, 'Portfolio_Volatility.png'))


# Mean of DSR
daily_mean_simple_returns = daily_simple_returns.mean()
print("Daily Mean Simple Returns: (percent)\n", daily_mean_simple_returns)


# Calculate expected portfolio daily return
portfolio_data['exp_roi(%)'] = (np.array(daily_mean_simple_returns) * portfolio_weights) * 253 * 100
portfolio_simple_return = np.sum(np.array(daily_mean_simple_returns) * portfolio_weights)


# Daily Returns
print("Expected daily returns on portfolio: " + str("{:.2f}".format(portfolio_simple_return * 100)) + " %")


# Annual Returns
print("Expected annual returns on portfolio: " + str("{:.2f}".format(portfolio_simple_return * 253 * 100)) + " %")


# Calculate Growth
daily_cumulative_simple_returns = (daily_simple_returns + 1).cumprod()


# Visualise daily cum simp returns
plt.figure(figsize=(12, 4))
for stock in daily_cumulative_simple_returns.columns.values:
    plt.plot(daily_cumulative_simple_returns.index, daily_cumulative_simple_returns[stock], lw=2, label=stock)

plt.legend(loc='upper left', fontsize=10)
plt.xlabel('Close')
plt.ylabel('Date')
plt.title("Daily Cumulative Simple Returns")
plt.savefig(os.path.join(_plots, 'Portfolio_DCSR_Distribution.png'))


# Save analysis highlights to a csv file
print(portfolio_data)
portfolio_data_analysed = os.path.join(_data, 'portfolio_analysed.csv')
portfolio_data.to_csv(portfolio_data_analysed)

plt.show()