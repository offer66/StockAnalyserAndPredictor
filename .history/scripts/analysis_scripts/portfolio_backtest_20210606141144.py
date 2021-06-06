import os
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError
import yfinance as yf

import matplotlib.pyplot as plt

import macd as macd
import rsi as rsi

_root = os.getcwd()
_portfolio = os.path.join(_root, 'datasets\\portfolio\\')
portfolio_data_raw = pd.read_csv(_portfolio + '\\portfolio_simple.csv')
portfolio_data = portfolio_data_raw

def run_backtest(stock, df, start_date, macd_dict, strat='rwb'):
	# exponential moving averages
	emas_used = [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]
	for x in emas_used:
		ema = x
		df["ema_" + str(ema)] = round(df.iloc[:, 4].ewm(span=ema, adjust=False).mean(), 2)

	# check if red, white, blue pattern
	pos, num, percent_change = 0, 0, []
	if strat == 'rwb':
		print("RWB Calculations...")
		for i in df.index:
			c_min = min(
				df['ema_3'][i].mean(),
				df['ema_5'][i].mean(),
				df['ema_8'][i].mean(),
				df['ema_10'][i].mean(),
				df['ema_12'][i].mean(),
				df['ema_15'][i].mean()
			)
			c_max = max(
				df['ema_30'][i].mean(),
				df['ema_35'][i].mean(),
				df['ema_40'][i].mean(),
				df['ema_45'][i].mean(),
				df['ema_50'][i].mean(),
				df['ema_60'][i].mean(),
			)
			close = df['Adj Close'][i]

			# Buying Condition
			if c_min > c_max:
				# print("Red White Blue")
				if pos == 0:		# we dont have anything
					buy_price = close
					pos = 1
					# print(f"Buying now at {str(buy_price)}")
				
			# Selling Position
			elif c_min < c_max:
				# print("Blue White Red")
				if pos == 1:		# we have a position
					pos = 0
					sell_price = close
					# print(f"Selling now at {str(sell_price)}")
					pc = 100 * (sell_price / buy_price - 1)
					percent_change.append(pc)

		# Closing all positions at the end of the date
		if (num == df['Adj Close'].count() - 1 and pos == 1):
			pos = 0
			sell_price = close
			# print(f"Selling now at {str(sell_price)}")
			pc = 100 * (sell_price / buy_price - 1)
			percent_change.append(pc)
	
		num += 1

		# print(percent_change)
		
		# Taking these historical results and analysing
		gains, net_gains, losses, net_losses, total_return = 0, 0, 0, 0, 1

		for i in percent_change:
			if i > 0:
				gains += i
				net_gains += 1
			else:
				losses += i
				net_losses += 1
			total_return = total_return * ((i / 100) + 1)

		total_return = round((total_return - 1) * 100, 2)

		if net_gains > 0:
			avg_gain = gains / net_gains
			max_return = str(max(percent_change))
		else:
			avg_gain = 0
			max_return = "Undefined"

		if net_losses > 0:
			avg_losses = losses / net_losses
			max_losses = str(min(percent_change))
			ratio = str(-avg_gain / avg_losses)
		else:
			avg_losses = 0
			max_losses = "Undefined"
			ratio = 'inf'
		if (net_gains > 0 or net_losses > 0):
			batting_avg = net_gains / (net_gains + net_losses)
		else:
			batting_avg = 0

		# Feedback
		print('------------------------------------------------------------------------------------------')
		print(f"RWB Results for {stock} going back to {start_date}, Sample Size: {str(net_gains + net_losses)} trades")
		# print(f'EMAs used: {str(emas_used)}')
		# print(f"Batting Avg: {str(batting_avg)}")
		# print(f"Gain/loss ratio: {ratio}")
		# print(f"Average Gain: {str(avg_gain)}")
		# print(f"Average Loss: {str(avg_losses)}")
		print(f"Max Return: {max_return}")
		print(f"Max Loss: {max_losses}")
		print(f"Total return over {str(net_gains + net_losses) } trades: {str(total_return)}%")
		print('------------------------------------------------------------------------------------------')
	

		return total_return, None

	elif strat == 'macd':
		print("MACD Calculations...")
		macd_dict[stock] = macd.DailyMACD(stock, start_date, 12, 26, 9)
		macd_dict[stock].run()
		buy, sell, profit = macd_dict[stock].get_buy_sell_profits()
		bd, sd = macd_dict[stock].get_buy_sell_dates()
		print('------------------------------------------------------------------------------------------')
		print(f"MACD Results for {stock} going back to {start_date}, Sample Size: {str(len(buy) + len(sell))} trades")
		# print("Buy Prices (USD): \n", buy)
		# print("Sell Prices (USD): \n", sell)
		# print("Gain/Loss (USD): \n", sell - buy)
		# print("Profits (%): \n", (sell - buy) / buy)
		print(f"Total Profits: {np.round(profit * 100, 2)} %")
		print('------------------------------------------------------------------------------------------')
		# print(macd_dict[stock].get_buy_sell_dates()[0], "\n", macd_dict[stock].get_buy_sell_dates()[1])
		# macd_dict[stock].view()

		return profit, None
	
	elif strat == 'rsi':
		print("RSI Calculations...")
		rsi_comparison = pd.DataFrame(columns=[
			"Company", "Current_RSI", "Days_Observed", "Crosses", 
			"True_Positive", "False_Positive", "True_Negative", "False_Negative", 
			"Sensitivity", "Specificity", "Accuracy", "TPR", "FPR"]
		)
		rsi_comparison, avg_gain, avg_loss = rsi.run(stock=stock, start_date=start_date, rsi_comparison=rsi_comparison)
		print("Note that 0% - 30% RSI is undervalued and 70% - 100% RSI is overvalued!")
		rsi_comparison = rsi_comparison.set_index('Company')
		print('------------------------------------------------------------------------------------------')
		print(f"RSI Results for {stock} going back to {start_date}")
		# print(f"Average Gain: {avg_gain} %")
		# print(f"Average Loss: {avg_loss} %")
		print(f"Total Profits: {avg_gain - avg_loss} %")
		print(f"RSI Value: {rsi_comparison.iloc[0]['Current_RSI']}")
		print(f"RSI Accuracy: {rsi_comparison.iloc[0]['Accuracy']}")
		if rsi_comparison.iloc[0]['TPR'] < rsi_comparison.iloc[0]['FPR']:
			print("RSI has given MORE False Positives than True Positives!!!")
			rsi_reliable = False
		else:
			print("RSI has given LESS False Positives than True Positives!!!")
			rsi_reliable = True
		return avg_gain - avg_loss, rsi_reliable


def run(symbols, backtest_time):
	today = dt.today().strftime('%Y-%m-%d')
	start_date = dt.today() - datetime.timedelta(days=backtest_time * 365)		# backtest_time years before today

	macd_dict = {}
	comparison_df = pd.DataFrame(columns=[
		'Symbol', 'RWB_Returns', 'MACD_Returns', 'RSI_Returns', 'RSI_Reliable'
	]
	)
	
	for ticker in symbols:
		strats = ['rwb', 'macd', 'rsi']

		print('------------------------------------------------------------------------------------------')
		print(f'{ticker}')
		print('------------------------------------------------------------------------------------------')
		row_add = {}
		strat_returns = []
		strat_returns.append(ticker)
		
		# GET DATA
		# yf.pdr_override()

		try:
			df = pdr.get_data_yahoo(ticker, start_date, today)
		except RemoteDataError:
			df = pdr.DataReader(f'{ticker}-USD','yahoo', start_date, today)

		print(df)
		for strat in strats:
			total_return, rsi_reliable = run_backtest(stock=ticker, df=df, start_date=start_date, macd_dict=macd_dict, strat=strat)
			strat_returns.append(total_return)
			if rsi_reliable is not None:
				strat_returns.append(rsi_reliable)

		i = 0
		for key in comparison_df.keys():
			row_add[key] = strat_returns[i]
			i += 1

		comparison_df = comparison_df.append(row_add, ignore_index=True)
	comparison_df = comparison_df.set_index('Symbol')
	print(comparison_df)
	comparison_df_analysed = os.path.join(_portfolio, 'portfolio_comparison_analysed.csv')
	comparison_df.to_csv(comparison_df_analysed)
	plt.show()

if __name__ == '__main__':
	stocks = ['AAPL', 'TSLA', 'AMD', 'ENPH', 'NIO', 'PLTR', 'ABNB', 'ETSY', 'GME']
	coins = ['BTC', 'ETH', 'DOGE', 'AAPL']
	portfolio = portfolio_data['symbol'].tolist()

	symbols = portfolio		# choose whether to use crypto or stocks or the portfolio csv

	backtest_time = 2		# will backtest from 2 years before today to today

	run(symbols=symbols, backtest_time=backtest_time)
