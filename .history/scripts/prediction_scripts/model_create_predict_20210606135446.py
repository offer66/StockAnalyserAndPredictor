import os
import datetime

# import IPython
# import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import pandas_market_calendars as mcal
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

import stock_scraper as pipeline
import model_comparison as validation_csv


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
	def __init__(self, symbols=['AAPL'], future=1, timescale='days', validate=False):
		# symbols to run ML on
		self.symbols = symbols	# 'AAPL', 'TSLA', 'GME', 'GOOG', 'ETSY', 'ENPH', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'

		# directories
		self._root = os.getcwd()
		self._data = os.path.join(self._root, 'datasets\\scraped\\')
		self._models = os.path.join(self._root, 'models')
		self._dirs = Variables.make_dirs(self)		# dict will be layed out according to --> {_dirs[ticker] = [_plots, _pred_data, _valid_data]}

		# data on dates and trading days
		self.today = datetime.date.today()
		self.start_date = self.today - datetime.timedelta(days=6 * 30)		# how far back we access yahoo finance stock EOD data
		self.valid_days = Variables.trading_days(self)

		# settings for ML
		self.future = future
		self.timescale = timescale
		self.validate = validate
		self.epochs = Variables.get_epoch(self)
		self.batch = Variables.get_batch(self)

		# dictionary with panda df's for each ticker in symbols
		self.data_dict = Variables.get_yahoo_data(self) if self.timescale == 'days' else Variables.get_minute_data(self)
		self.subsampled_data = Variables.subsample(self)		# dict will be layed out according to --> {ticker: [dataframe, date_time, timestamps]}

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
		epoch_dict = {'mins': 10, 'hours': 50, 'days': 100}
		epoch = epoch_dict[self.timescale]
		return epoch

	def get_batch(self):
		# if timescale is minutes then batch for hour trends (60mins in an hour)
		# if timescale is hours then batch for day trends (16hrs in a trading day)
		scales = {'mins': 60, 'hours': 16, 'days': 5, 'weeks': 4}
		batch = scales[self.timescale]
		return batch

	def get_minute_data(self):
		data_dict = {ticker: [] for ticker in self.symbols}
		for ticker in data_dict:
			data_dict[ticker] = pd.read_csv(os.path.join(os.path.join(self._data, ticker), f'{ticker}-total-data.csv'))
			print(f"{ticker} yahoo data: \n {data_dict[ticker].head(3)} \n ---------- \n {data_dict[ticker].tail(3)} \n ---------- \n {data_dict[ticker].shape}")
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
			print(f"{ticker} yahoo data: \n {data.head(3)} \n ---------- \n {data.tail(3)} \n ---------- \n {data.shape}")
		return data_dict

	def subsample(self):
		# dict will be layed out according to --> {ticker: [dataframe, date_time, timestamps]}
		subsampled_data = {ticker: [] for ticker in self.symbols}
		for ticker in subsampled_data.keys():
			dataframe = self.data_dict[ticker]
			dataframe = dataframe.reset_index()
			dataframe = dataframe[['time', 'close', 'open', 'high' , 'low', 'volume']]
			if self.timescale == 'days' or self.timescale == 'mins':
				print("HERE: \n", dataframe)
				date_time = pd.to_datetime(dataframe.pop('time'), format='%Y-%m-%d %H:%M:%S')
				timestamps = date_time.map(datetime.datetime.timestamp)
			elif self.timescale == 'hours':
				dataframe['time'] = pd.to_datetime(dataframe['time'], format='%Y.%m.%d %H:%M:%S')

				# this only uses data that has the timestamp HH:00:00
				dataframe = dataframe.set_index('time')
				dataframe = dataframe[dataframe.index.minute == 0]
				dataframe = dataframe.reset_index('time')

				date_time = pd.to_datetime(dataframe.pop('time'), format='%Y-%m-%d %H:%M:%S')
				timestamps = date_time.map(datetime.datetime.timestamp)
			
			subsampled_data[ticker] = [dataframe, date_time, timestamps]
		return subsampled_data

	def create_future_dates(self):
		date_time = self.subsampled_data[symbols[0]][1]		# this is assuming all tickers trade on the same day
		if self.timescale == 'days':
			next_dates = Variables.next_trading_days(self)
		
		print(next_dates)
		total_dates = date_time.tolist() if not isinstance(date_time, list) else date_time
		for date in next_dates:
			total_dates.append(date)
		return total_dates, next_dates

	def next_trading_days(self):
		nyse = mcal.get_calendar('NYSE')
		trading_days = nyse.valid_days(start_date=self.last_date, end_date=self.last_date + datetime.timedelta(days=6 * 30))
		formatted_days = []
		for date in trading_days:
			formatted_days.append(date.strftime('%Y-%m-%d %H:%M:%S'))
		return formatted_days[1 : self.future + 1]



def model_exists(path_vars):
	symbol, shift, timescale = path_vars[0], path_vars[1], path_vars[2]
	model_path = os.path.join(_models, f'{symbol}-{timescale}-{var.epochs}epochs')		# currently checking if multi variable models exist
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
	print('Test RMSE: %.3f' % rmse)

	pred_data = pd.DataFrame({
		'time': [], 'close': [], 'open': [],
		'high': [], 'low': [], 'volume': [],
	})
	actual_data = pd.DataFrame({
		'time': [], 'close': [], 'open': [],
		'high': [], 'low': [], 'volume': [],
	})
	valid_plot = pd.DataFrame({
		'time': [], 'close': [], 'predicted': []
	})
	
	# print(len(test_dates), inv_yhat[:,0].shape, inv_y[:,0].shape)
	# pred_data['time'], actual_data['time'] = date_time.tolist()[int(n*0.7)-8:-batch-8], date_time.tolist()[int(n*0.7)-8:-batch-8]
	pred_data['time'], actual_data['time'] = date_time.tolist()[int(n*0.7)+batch:], date_time.tolist()[int(n*0.7)+batch:]
	
	pred_data['close'], pred_data['open'], pred_data['high'], pred_data['low'], pred_data['volume'] = inv_yhat[:,0], inv_yhat[:,1], inv_yhat[:,2], inv_yhat[:,3], inv_yhat[:,4]
	actual_data['close'], actual_data['open'], actual_data['high'], actual_data['low'], actual_data['volume'] = inv_y[:,0], inv_y[:,1], inv_y[:,2], inv_y[:,3], inv_y[:,4]
	
	pred_data = pred_data.drop_duplicates(subset=['time'])
	actual_data = actual_data.drop_duplicates(subset=['time'])

	valid_plot['time'], valid_plot['close'], valid_plot['predicted'] = pred_data['time'], actual_data['close'], pred_data['close']

	pred_data.set_index('time', inplace=True)
	actual_data.set_index('time', inplace=True)
	valid_plot.set_index('time', inplace=True)

	# pred_data = pred_data.shift(batch)
	# actual_data = actual_data.shift(batch+future)
	
	# plot
	fig2, ax2 = plt.subplots()
	ax2.set_title(f'{symbol} Predicted Prices for {var.timescale} from historical data: ')
	ax2.set_xlabel(f'Dates')
	ax2.set_ylabel(f'Price (USD)')
	ax2.plot(valid_plot[['close', 'predicted']])
	ax2.legend(['Actual', 'Predicted'], loc='lower right')
	# plt.show()
	return pred_data, actual_data, valid_plot
	

def prediction_loop(input_data, input_objs, input_vars):
	x_total, predicted_data, df = input_data
	scaler, model = input_objs
	batch, num_features = input_vars

	test_data = x_total[-1: ]		# latest data point
	yhat = model.predict(test_data)

	test_data = test_data.reshape((test_data.shape[0], batch*num_features))
	test_data = np.concatenate((yhat, test_data[:, 1-num_features:]), axis=1)
	# print("shape for inverse transform \n", test_data.shape)

	predicted_data['close'], predicted_data['open'], predicted_data['high'], predicted_data['low'], predicted_data['volume'] = scaler.inverse_transform(test_data)[:,0], scaler.inverse_transform(test_data)[:,1], scaler.inverse_transform(test_data)[:,2], scaler.inverse_transform(test_data)[:,3], scaler.inverse_transform(test_data)[:,4]
	# print("\n predicted df \n", predicted_data)
	df = pd.concat([df, predicted_data])
	df = df.set_index(['time'])
	# print("\n combined df \n", df)

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

	df = var.subsampled_data[symbol][0]
	date_time = var.subsampled_data[symbol][1]
	timestamps = var.subsampled_data[symbol][2]

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
	import tensorflow as tf
	if train_model:
		model = tf.keras.Sequential([
			tf.keras.layers.LSTM(units=50, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])),
			tf.keras.layers.Dense(units=1)
		])
		## this model for long long epoch training (currently dont have the data for that to be useful)
		# model = tf.keras.Sequential([
		# 	tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
		# 	tf.keras.layers.Dropout(0.2),
		# 	tf.keras.layers.LSTM(units=50, return_sequences=True),
		# 	tf.keras.layers.Dropout(0.2),
		# 	tf.keras.layers.LSTM(units=50, return_sequences=True),
		# 	tf.keras.layers.Dropout(0.2),
		# 	tf.keras.layers.LSTM(units=50, return_sequences=False),
		# 	tf.keras.layers.Dropout(0.2),
		# 	tf.keras.layers.Dense(units=1)
		# ])
		# train the model
		model.compile(optimizer='adam', loss='mean_squared_error')
		history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch, validation_data=(x_test, y_test), verbose=1, shuffle=True)
		# save the model
		model.save(os.path.join(_models, f'{symbol}-{timescale}-{epochs}epochs'))
		# quick plot of train losses vs test losses
		fig1, ax1 = plt.subplots()
		ax1.plot(history.history['loss'], label='train')
		ax1.plot(history.history['val_loss'], label='test')
		ax1.set_xlabel(f'Number of Epochs')
		ax1.set_ylabel(f'Loss Value')
		ax1.set_title(f'{symbol} train loss vs test loss: ')


	## verify model performance compared to actual data
	model = tf.keras.models.load_model(os.path.join(_models, f'{symbol}-{timescale}-{epochs}epochs'))
	if not predict_future or validate:
		print("Verifying Models Predictions...")
		input_data = [x_test, date_time, y_test]
		input_objs = [model, scaler]
		input_vars = [symbol, batch, num_features, n]
		pred_data, actual_data, valid_plot = verify_model_predictions(input_data, input_objs, input_vars)
		print("Predicted DF: \n", pred_data)
		print("Actual DF: \n", actual_data)

		# add to validation csv (to compare between models)
		valid_file_name = f'{symbol}-MV-manual-data-{timescale}-{epochs}-epochs-validation.csv'
		valid_plot.to_csv(os.path.join(var._dirs[symbol][2], valid_file_name))
		validation_csv.run(valid_file_name, symbol)

	## make new predictions 
	else:
		print("Predicting New Close Prices...")
		predicted_data = pd.DataFrame({'time': [], 'close': [], 'open': [], 'high': [], 'low': [], 'volume': []})

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

		df['time'] = total_dates
		df = df.set_index(['time'])
		actual_df['time'] = total_dates[:-var.future]
		actual_df = actual_df.set_index(['time'])

		# plot new predictions
		fig3, ax3 = plt.subplots()
		ax3.set_title(f'{symbol}s Historical Data and {var.future} New Predicted Prices: ')
		ax3.set_xlabel(f'Dates')
		ax3.set_ylabel(f'Price (USD)')
		ax3.plot(df['close'], marker='x')
		ax3.plot(actual_df['close'])
		ax3.legend(['predicted', 'actual'], loc='lower right')

		print("\n prediction data (last 3 values): \n", df.tail(3))
		print("\n actual data (last 3 values): \n", actual_df.tail(3))


if __name__=="__main__":
	# 'AAPL', 'TSLA', 'GME', 'ABNB', 'PLTR', 'ETSY', 'ENPH', 'GOOG', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'
	symbols = [
		'AAPL'
	]

	# class that initialises our variables, data, objs
	var = Variables(symbols=symbols, future=1, timescale='mins', validate=True)

	for symbol in var.symbols:
		# define training and predict bools based on if a model already exists
		train_bool = False if model_exists(path_vars=[symbol, var.future, var.timescale]) else True
		predict_future = False if train_bool else True

		# run the training/predicting script
		create_and_predict(symbol=symbol)

	# show our plotted graphs
	plt.show()