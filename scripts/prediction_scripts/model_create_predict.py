import os
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

import tensorflow as tf

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



class Variables:
	def __init__(self, symbols: list=['AAPL'], future: int=1, timescale: str='days', validate: bool=False, extra_cols_bool: bool=False):
		print(f"Gathering data for {symbols} and creating variables for training...")
		# symbols to run ML on
		self.symbols = symbols	# 'AAPL', 'TSLA', 'GME', 'GOOG', 'ETSY', 'ENPH', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'

		# directories
		self._data = os.path.join(_root, 'datasets\\scraped\\')
		self._dirs = Variables.make_dirs(self)		# dict will be layed out according to --> {_dirs[ticker] = [_plots, _pred_data, _valid_data]}
		
		# data on dates and trading days
		self.today = datetime.date.today()
		self.start_date = self.today - datetime.timedelta(days=6 * 30)		# how far back we access yahoo finance stock EOD data
		self.valid_days = Variables.trading_days(self)
		
		# settings for ML
		self.future = future
		self.timescale = timescale
		self.validate = validate
		self.extra_cols_bool = extra_cols_bool
		self.epochs = Variables.get_epoch(self)
		self.batch = Variables.get_batch(self)
		
		# dictionary with panda df's for each ticker in symbols
		# dict will be layed out according to --> {ticker: [dataframe, date_time]}
		self.data_dict = Variables.get_yahoo_data_faster(self) if self.timescale == 'days' else Variables.get_minute_data(self)
		self.subsampled_data = Variables.subsample(self)
		self.extra_cols_data = Variables.extra_cols_calculations(self)
		self.column_names = self.extra_cols_data[self.symbols[0]][0].columns if self.extra_cols_bool else self.subsampled_data[self.symbols[0]][0].columns		# assuming all df's have the same col names

		# obtains the last trading date by accessing the last documented date for the first symbols data df
		self.last_date = self.subsampled_data[symbols[0]][1].iat[-1]

		# sorting through future dates
		self.total_dates, self.next_dates = Variables.create_future_dates(self)

	def make_dirs(self):
		# dict will be layed out according to --> {_dirs[ticker] = [_plots, _pred_data, _valid_data]}
		_dirs = {ticker: [] for ticker in self.symbols}
		for ticker in _dirs:
			_plots = os.path.join(_root, f'plots\\{ticker}')
			_pred_data = os.path.join(_root, f'datasets\\predicted\\{ticker}')
			_valid_data = os.path.join(_root, f'datasets\\validation\\{ticker}')
			try:
				os.mkdir(_plots)
			except:
				pass
			try:
				os.mkdir(os.path.join(_plots, 'Predictions'))
			except:
				pass
			try:
				os.mkdir(_pred_data)
			except:
				pass
			try:
				os.mkdir(_valid_data)
			except:
				pass
			_dirs[ticker] = [_plots, _pred_data, _valid_data]
		return _dirs

	def trading_days(self):
		# gets a list of valid trading days (makes it easier when quickly needing the next trading day such as a Friday or bank holiday)
		nyse = mcal.get_calendar('NYSE')

		# gets data in range 365*2 (all trading days between past 2 years and future 2 years)
		start_date = self.today - datetime.timedelta(days=365 * 2)
		end_date = self.today + datetime.timedelta(days=365 * 2)
		valid_days = nyse.valid_days(start_date=start_date, end_date=end_date)
		return valid_days

	def get_epoch(self):
		epoch_dict = {'mins': 2, 'days': 100}
		epoch = epoch_dict[self.timescale]
		return epoch

	def get_batch(self):
		# if timescale is minutes then batch for hour trends (60mins in an hour)
		scales = {'mins': 60, 'days': 5, 'weeks': 4}
		batch = scales[self.timescale]
		return batch

	def get_minute_data(self):
		data_dict = {ticker: [] for ticker in self.symbols}
		for ticker in data_dict:
			data_dict[ticker] = pd.read_csv(os.path.join(os.path.join(self._data, ticker), f'{ticker}-total-data.csv'))
			# print(f"{ticker} yahoo data: \n {data_dict[ticker].head(3)} \n ---------- \n {data_dict[ticker].tail(3)} \n ---------- \n {data_dict[ticker].shape}")
		return data_dict

	def get_yahoo_data(self):
		data_dict = {ticker: [] for ticker in self.symbols}
		for ticker in data_dict:
			data = pdr.get_data_yahoo(ticker, start=self.start_date, end=self.today)
			data = data.reset_index()
			data.columns = ['time', 'open', 'high', 'low', 'close', 'adj close', 'volume']
			data = data[['time', 'open', 'high', 'low', 'close', 'volume']]
			data = data.set_index('time')
			data_dict[ticker] = data
			# print(f"{ticker} yahoo data: \n {data.head(3)} \n ---------- \n {data.tail(3)} \n ---------- \n {data.shape}")
		return data_dict
	
	def get_yahoo_data_faster(self):	# faster scraping method (does all tickers at once)
		data = pdr.get_data_yahoo(self.symbols, start=self.start_date, end=self.today)
		data = data.swaplevel(0, 1, axis=1)
		data = data.reset_index()
		data = data.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj close', 'Volume': 'volume'})
		data = data.set_index('time')
		return data

	def subsample(self):
		# dict will be layed out according to --> {ticker: [dataframe, date_time]}
		subsampled_data = {ticker: [] for ticker in self.symbols}
		for ticker in subsampled_data.keys():
			dataframe = self.data_dict[ticker]
			dataframe = dataframe.reset_index()
			dataframe = dataframe[['time', 'close', 'open', 'high' , 'low', 'volume']]
			if self.timescale == 'days':
				date_time = pd.to_datetime(dataframe.pop('time'))
			elif self.timescale == 'mins':
				date_time = dataframe.pop('time')
			subsampled_data[ticker] = [dataframe, date_time]
		return subsampled_data
		
	def extra_cols_calculations(self):
		extra_cols_dataframe = {ticker: [] for ticker in self.subsampled_data.keys()}
		for ticker in self.subsampled_data.keys():
			dataframe = self.subsampled_data[ticker][0]
			## do things to dataframe to get additional columns
			# emas
			emas_used = [3, 5, 10, 12, 26, 30]
			smas_used = [10, 30, 50]
			for x in emas_used:
				ema = x
				dataframe["ema_" + str(ema)] = round(dataframe.iloc[:, 0].ewm(span=ema, adjust=False).mean(), 2)
			# smas
			for x in smas_used:
				sma = x
				dataframe["sma_" + str(sma)] = round(dataframe.iloc[:, 0].ewm(span=sma, adjust=False).mean(), 2)
			# macd
			dataframe["macd"] = dataframe['ema_12'] - dataframe['ema_26']
			# rsi
			up_prices, down_prices = rsi_calc.up_down(dataframe['close'].values)
			avg_gain, avg_loss = rsi_calc.averages(up_prices, down_prices)
			RS, RSI = rsi_calc.rsi(dataframe['close'].values, avg_gain, avg_loss)
			dataframe["rsi"] = RSI
			# social media sentiment

			## add the new dataframe to extra_cols_dataframe
			dataframe = dataframe[['close', 'open', 'high', 'low', 'volume', 'ema_3', 'ema_5', 'ema_10', 'ema_30', 'sma_10', 'sma_30', 'sma_50', 'macd', 'rsi']]
			extra_cols_dataframe[ticker] = [dataframe, self.subsampled_data[ticker][1]]
		return extra_cols_dataframe

	def create_future_dates(self):
		date_time = self.subsampled_data[symbols[0]][1]		# this is assuming all tickers trade on the same day
		next_dates = Variables.next_trading_days(self)
		
		total_dates = date_time.tolist() if not isinstance(date_time, list) else date_time
		for date in next_dates:
			total_dates.append(date)
		return total_dates, next_dates

	def next_trading_days(self):
		nyse = mcal.get_calendar('NYSE')
		new_dates = []
		# grab 6 months worth of future trading days to then find the needed dates within this list
		self.last_date = pd.to_datetime(self.last_date)
		trading_days = nyse.valid_days(start_date=self.last_date + datetime.timedelta(days=1), end_date=self.last_date + datetime.timedelta(days=6 * 30))
		if self.timescale == 'days':
			for date in trading_days:
				new_dates.append(date.strftime('%Y-%m-%d %H:%M:%S'))
			new_dates = new_dates[: self.future]
			return new_dates
		elif self.timescale == 'mins':
			# checking to see if we are predicting less than a trading day ahead
			if self.future < 390:
				schedule = nyse.schedule(start_date=trading_days[0], end_date=trading_days[0])
				diff = 390 - self.future	# how far off our prediction is to the end of the actual day
			else:
				# if more than a full day then get self.future in num of days and round up
				schedule = nyse.schedule(start_date=trading_days[0], end_date=trading_days[int(self.future/390)])
				diff = (1 + int(self.future / 390))*390 - self.future	# how far off our prediction is to the end of the actual day
			# create a list of datetime objects for the schedule times
			schedule.index = pd.to_datetime(schedule.index)
			for date in list(schedule.index.values):
				if date == list(schedule.index.values)[-1]:
					mins_difference = 390-diff
				else:
					mins_difference = 390
				for i in range(mins_difference):
					transformed_date = schedule.loc[date]['market_open'] + datetime.timedelta(minutes=i)
					transformed_date = pd.to_datetime(transformed_date)
					new_dates.append((transformed_date.strftime('%Y-%m-%d %H:%M:%S')))
			return new_dates



def model_exists(path_vars):
	symbol, shift, timescale, model_name = path_vars[0], path_vars[1], path_vars[2], path_vars[3]
	model_path = os.path.join(_models, model_name)		# currently checking if multi variable models exist
	return os.path.exists(model_path)
	

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def verify_model_predictions(input_data, input_objs, input_vars):
	x_test, date_time, y_test = input_data
	model, scaler = input_objs
	symbol, batch, num_features, n = input_vars

	yhat = model.predict(x_test)
	x_test = x_test.reshape((x_test.shape[0], batch*num_features))
	
	x_test_and_pred = np.concatenate((yhat, x_test[:, 1-num_features:]), axis=1)
	inv_yhat = scaler.inverse_transform(x_test_and_pred)
	
	y_test = y_test.reshape((len(y_test), 1))
	inv_y = np.concatenate((y_test, x_test[:, 1-num_features:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	
	rmse = np.sqrt(mean_squared_error(inv_y[:,0], inv_yhat[:,0]))

	predicted_data = pd.DataFrame({name: [] for name in var.column_names})
	actual_data = pd.DataFrame({name: [] for name in var.column_names})

	# predicted_data = pd.DataFrame({
	# 	'time': [], 'close': [], 'open': [],
	# 	'high': [], 'low': [], 'volume': [],
	# })
	# actual_data = pd.DataFrame({
	# 	'time': [], 'close': [], 'open': [],
	# 	'high': [], 'low': [], 'volume': [],
	# })

	valid_plot = pd.DataFrame({
		'time': [], 'close': [], 'predicted': []
	})
	
	predicted_data['time'], actual_data['time'] = date_time.tolist()[int(n*0.7)+batch:], date_time.tolist()[int(n*0.7)+batch:]

	i = 0
	for name in var.column_names:
		predicted_data[name] = inv_yhat[:,i]
		actual_data[name] = inv_y[:,i]
		i += 1

	# predicted_data['close'], predicted_data['open'], predicted_data['high'], predicted_data['low'], predicted_data['volume'] = inv_yhat[:,0], inv_yhat[:,1], inv_yhat[:,2], inv_yhat[:,3], inv_yhat[:,4]
	# actual_data['close'], actual_data['open'], actual_data['high'], actual_data['low'], actual_data['volume'] = inv_y[:,0], inv_y[:,1], inv_y[:,2], inv_y[:,3], inv_y[:,4]
	
	predicted_data = predicted_data.drop_duplicates(subset=['time'])
	actual_data = actual_data.drop_duplicates(subset=['time'])

	valid_plot['time'], valid_plot['close'], valid_plot['predicted'] = pd.to_datetime(predicted_data['time']), actual_data['close'], predicted_data['close']

	predicted_data.set_index('time', inplace=True)
	actual_data.set_index('time', inplace=True)
	valid_plot.set_index('time', inplace=True)

	# plot
	fig2, ax2 = plt.subplots()
	ax2.set_title(f'{symbol} Predicted Prices for {var.timescale} from historical data: ')
	ax2.set_xlabel(f'Dates')
	ax2.set_ylabel(f'Price (USD)')
	splits = int(valid_plot.shape[0]/100) if valid_plot.shape[0] > 100 else 1
	ax2.plot(valid_plot[['close', 'predicted']][::splits])
	ax2.legend(['Actual', 'Predicted'], loc='lower right')
	# plt.show()
	return predicted_data, actual_data, valid_plot
	

def prediction_loop(input_data, input_objs, input_vars):
	x_total, predicted_data, df = input_data
	scaler, model = input_objs
	batch, num_features = input_vars

	test_data = x_total[-1: ]		# latest data point
	yhat = model.predict(test_data)

	test_data = test_data.reshape((test_data.shape[0], batch*num_features))
	test_data = np.concatenate((yhat, test_data[:, 1-num_features:]), axis=1)
	# print("shape for inverse transform \n", test_data.shape)

	i = 0
	for name in var.column_names:
		predicted_data[name] = scaler.inverse_transform(test_data)[:,i]
		i += 1

	# predicted_data['close'], predicted_data['open'], predicted_data['high'], predicted_data['low'], predicted_data['volume'] = scaler.inverse_transform(test_data)[:,0], scaler.inverse_transform(test_data)[:,1], scaler.inverse_transform(test_data)[:,2], scaler.inverse_transform(test_data)[:,3], scaler.inverse_transform(test_data)[:,4]
	
	# print("\n predicted df \n", predicted_data)
	df = pd.concat([df, predicted_data])
	try:
		df = df.set_index(['time'])
	except KeyError:
		pass

	return predicted_data, df


def transform_data(dataframe, n, num_features):
	# redefine class variables for the function
	date_time = var.subsampled_data[symbol][1]
	batch = var.batch

	# transform and scale dataframe values
	values = dataframe.values
	values = values.astype('float32')

	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_values = scaler.fit_transform(values)

	total_df = series_to_supervised(data=scaled_values, n_in=batch)

	# split the data into train and test values
	total_values = total_df.values
	train_values = total_values[: int(n*0.7), :]
	test_values = total_values[int(n*0.7):, : ]
	
	# split the train and test values into x and y (x is the input and y is the expected output)
	x_train, y_train = train_values[:, :batch*num_features], train_values[:, -num_features]
	x_train = x_train.reshape((x_train.shape[0], batch, num_features))

	x_test, y_test = test_values[:, :batch*num_features], test_values[:, -num_features]
	x_test = x_test.reshape((x_test.shape[0], batch, num_features))

	x_total, y_total = total_values[:, :batch*num_features], total_values[:, -num_features]
	x_total = x_total.reshape((x_total.shape[0], batch, num_features))
	
	return x_train, y_train, x_test, y_test, x_total, y_total, scaler


def create_and_predict(symbol):
	## setting all the variables
	symbol = symbol

	df = var.extra_cols_data[symbol][0] if var.extra_cols_bool else var.subsampled_data[symbol][0]
	date_time = var.extra_cols_data[symbol][1] if var.extra_cols_bool else var.subsampled_data[symbol][1]
	model_name = f'{symbol}-{var.timescale}-{var.epochs}epochs-extracol' if var.extra_cols_bool else f'{symbol}-{var.timescale}-{var.epochs}epochs'
	valid_file_name = f'{symbol}-{var.timescale}-{var.epochs}epochs-extracol-validation.csv' if var.extra_cols_bool else f'{symbol}-{var.timescale}-{var.epochs}epochs-validation.csv'

	timescale = var.timescale
	epochs = var.epochs
	train_model = train_bool
	validate = var.validate
	batch = var.batch

	n = df.shape[0]
	num_features = df.shape[1]
	last_date = date_time.iloc[-1]

	total_dates = var.total_dates
	next_dates = var.next_dates


	## transform our data and get data for training and plotting
	x_train, y_train, x_test, y_test, x_total, y_total, scaler = transform_data(dataframe=df, n=n, num_features=num_features)


	## define and train the model
	if train_model:
		model = tf.keras.Sequential([
			tf.keras.layers.LSTM(units=50, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])),
			tf.keras.layers.Dense(units=1)
		])
		# train the model
		model.compile(optimizer='adam', loss='mean_squared_error')
		history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch, validation_data=(x_test, y_test), verbose=1, shuffle=True)
		# save the model
		model.save(os.path.join(_models, model_name))
		# quick plot of train losses vs test losses
		fig1, ax1 = plt.subplots()
		ax1.plot(history.history['loss'], label='train')
		ax1.plot(history.history['val_loss'], label='test')
		ax1.set_xlabel(f'Number of Epochs')
		ax1.set_ylabel(f'Loss Value')
		ax1.set_title(f'{symbol} train loss vs test loss: ')


	## verify model performance compared to actual data
	model = tf.keras.models.load_model(os.path.join(_models, model_name))
	if not predict_future or validate:
		print("Verifying Models Predictions...")
		input_data = [x_test, date_time, y_test]
		input_objs = [model, scaler]
		input_vars = [symbol, batch, num_features, n]
		predicted_data, actual_data, valid_plot = verify_model_predictions(input_data, input_objs, input_vars)

		# add to validation csv (to compare between models)
		valid_plot.to_csv(os.path.join(var._dirs[symbol][2], valid_file_name))
		validation_csv.run(valid_file_name, symbol)

	## make new predictions 
	else:
		print("Predicting New Close Prices...")
		predicted_data = pd.DataFrame({name: [] for name in var.column_names})

		actual_df = df.copy()
		for i in range(var.future):
			if i != 0:
				total_values = series_to_supervised(data=scaler.fit_transform(df.values), n_in=batch).values
				x_total_test = total_values[:, :batch*num_features]
				x_total = x_total_test.reshape((x_total_test.shape[0], batch, num_features))[-1:]
			input_data = [x_total, predicted_data, df]
			input_objs = [scaler, model]
			input_vars = [batch, num_features]
			prediction_data, df = prediction_loop(input_data, input_objs, input_vars)
		
		df['time'] = pd.to_datetime(total_dates)
		df = df.drop_duplicates(subset=['time'])
		df = df.set_index(['time'])

		actual_df['time'] = pd.to_datetime(total_dates[:-var.future])
		actual_df = actual_df.drop_duplicates(subset=['time'])
		actual_df = actual_df.set_index(['time'])

		# plot new predictions
		fig3, ax3 = plt.subplots()
		ax3.set_title(f'{symbol}s Historical Data and {var.future} New Predicted Prices: ')
		ax3.set_xlabel(f'Dates')
		ax3.set_ylabel(f'Price (USD)')
		splits = int(df.shape[0]/100) if df.shape[0] > 100 else 1
		ax3.plot(df['close'][int(0.5*df.shape[0])::splits], marker='x')
		ax3.plot(actual_df['close'][int(0.5*df.shape[0])::splits])
		ax3.legend(['predicted', 'actual'], loc='lower right')

		df.to_csv(os.path.join(var._dirs[symbol][1], valid_file_name))
		
		print(f"\n prediction data (last {5 + var.future} values): \n {df.tail(5 + var.future)}")
		print(f"\n actual data (last 5 values): \n {actual_df.tail(5)}")


if __name__=="__main__":
	# 'AAPL', 'GME', 'ABNB', 'PLTR', 'ETSY', 'ENPH', 'GOOG', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'
	# 'BTC-USD', 'ETH-USD', 'NANO-USD', 'ADA-USD'
	symbols = [
		'AAPL', 'GME', 'ABNB', 'PLTR', 'ETSY', 'ENPH', 'GOOG', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'
	]

	# class that initialises our variables, data, objs
	var = Variables(symbols=symbols, future=1, timescale='days', validate=False, extra_cols_bool=True)
	
	for symbol in var.symbols:
		# define training and predict bools based on if a model already exists
		model_name = f'{symbol}-{var.timescale}-{var.epochs}epochs-extracol' if var.extra_cols_bool else f'{symbol}-{var.timescale}-{var.epochs}epochs'
		train_bool = False if model_exists(path_vars=[symbol, var.future, var.timescale, model_name]) else True
		predict_future = False if train_bool else True
		
		# run the training/predicting script
		create_and_predict(symbol=symbol)

	# show our plotted graphs
	# plt.show()