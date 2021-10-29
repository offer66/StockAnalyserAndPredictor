import os
import time
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts/analysis_scripts"))
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import pandas_market_calendars as mcal
import yfinance as yf

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import model_validation as validation_csv
import model_comparison as comparison_csv
import stock_scraper_local as scraper

import rsi as rsi_calc

# import initialisations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
yf.pdr_override()

mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["axes.grid"] = False

# some base directories
_root = os.getcwd()
_data = os.path.join(_root, "datasets\\scraped\\")
_models = os.path.join(_root, "models")


class InitialiseDirs:
    """Initialises and creates the data and dir directories"""

    def __init__(self, symbols):
        self.symbols = symbols
        self._dirs = self.make_dirs()

    def make_dirs(self) -> list:
        # dict will be layed out according to --> {_dirs[ticker] = [_plots, _pred_data, _valid_data]}
        _dirs = {ticker: [] for ticker in self.symbols}
        for ticker in _dirs:
            _plots = os.path.join(_root, f"datasets\\plots\\{ticker}")
            _pred_data = os.path.join(_root, f"datasets\\predicted\\{ticker}")
            _valid_data = os.path.join(_root, f"datasets\\validation\\{ticker}")
            _scraped_data = os.path.join(_root, f"datasets\\scraped\\{ticker}")
            try:
                os.mkdir(_plots)
                os.mkdir(os.path.join(_plots, "Predictions"))
                os.mkdir(_pred_data)
                os.mkdir(_valid_data)
            except FileExistsError:
                pass
            _dirs[ticker] = [_plots, _pred_data, _valid_data, _scraped_data]
        return _dirs


class InitialiseDates:
    """Initialises today, start date and valid trading days"""

    def __init__(
        self,
        start=datetime.date.today() - datetime.timedelta(days=6 * 30),
        end=datetime.date.today(),
    ):
        self.today = end
        self.start_date = start  # how far back we access yahoo finance stock EOD data
        self.valid_days = self.trading_days()

    def trading_days(self) -> list:
        """gets a list of valid trading days for past and future 2 years"""
        nyse = mcal.get_calendar("NYSE")
        start_date = self.today - datetime.timedelta(days=365 * 2)
        end_date = self.today + datetime.timedelta(days=365 * 2)
        valid_days = nyse.valid_days(start_date=start_date, end_date=end_date)
        return valid_days


class InitialiseMLVars:
    """Initialises the variables needed for the ML training"""

    def __init__(
        self,
        future: int = 1,
        timescale: str = "days",
        extra_cols_bool: bool = True,
    ):
        self.future = future
        self.timescale = timescale
        self.extra_cols_bool = extra_cols_bool
        self.epochs = self.get_epoch()
        self.batch = self.get_batch()

    def get_epoch(self) -> int:
        epoch_dict = {"mins": 2, "days": 100}
        epoch = epoch_dict[self.timescale]
        return epoch

    def get_batch(self) -> int:
        # if timescale is minutes then batch for hour trends (60mins in an hour)
        scales = {"mins": 60, "days": 5, "weeks": 4}
        batch = scales[self.timescale]
        return batch


class StockData:
    def __init__(self, dates, ML, symbols: list = ["AAPL"]):
        self.symbols: list = symbols
        self._dirs = InitialiseDirs(self.symbols)._dirs
        self.start_date = dates.start_date
        self.today = dates.today
        self.timescale: str = ML.timescale
        self.future: int = ML.future
        self.extra_cols_bool: bool = ML.extra_cols_bool

        self.data_dict: dict = (
            self.get_yahoo_data()
            if self.timescale == "days"
            else self.get_minute_data()
        )
        self.subsampled_data: dict = self.subsample()
        self.extra_cols_data: dict = self.extra_cols_calculations()
        self.column_names: list = (
            self.extra_cols_data[self.symbols[0]][0].columns
            if self.extra_cols_bool
            else self.subsampled_data[self.symbols[0]][0].columns
        )  # assuming all df's have the same col names

        self.last_date = self.subsampled_data[symbols[0]][1].iat[-1]
        self.total_dates, self.next_dates = self.create_future_dates()

    def get_minute_data(self) -> dict:
        data_dict = {ticker: [] for ticker in self.symbols}
        for ticker in data_dict:
            data_dict[ticker] = pd.read_csv(
                os.path.join(self._dirs[ticker][3], f"{ticker}-total-data.csv")
            )
            # print(
            #     f"{ticker} yahoo data: \n {data_dict[ticker].head(3)} \n ---------- \n {data_dict[ticker].tail(3)} \n ---------- \n {data_dict[ticker].shape}"
            # )
        return data_dict

    def get_yahoo_data(
        self,
    ) -> pd.DataFrame:  # faster scraping method (does all tickers at once)
        data = pdr.get_data_yahoo(self.symbols, start=self.start_date, end=self.today)
        try:
            data = data.swaplevel(0, 1, axis=1)
        except TypeError:
            pass  # if TypeError then it's not a multindexed dataframe
        data = data.reset_index()
        data = data.rename(
            columns={
                "Date": "time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj close",
                "Volume": "volume",
            }
        )
        data = data.set_index("time")
        return data

    def subsample(self) -> dict:
        """subsamples the data depending on timeframes (eg. splits into hour only data)"""
        # dict --> {ticker: [dataframe, date_time]}
        subsampled_data = {ticker: [] for ticker in self.symbols}
        for ticker in subsampled_data.keys():
            try:
                dataframe = self.data_dict[ticker]
            except KeyError:
                dataframe = self.data_dict  # just a single df and not a dict
            dataframe = dataframe.reset_index()
            dataframe = dataframe[["time", "close", "open", "high", "low", "volume"]]
            if self.timescale == "days":
                date_time = pd.to_datetime(dataframe.pop("time"))
            elif self.timescale == "mins":
                date_time = dataframe.pop("time")
            subsampled_data[ticker] = [dataframe, date_time]
        return subsampled_data

    def extra_cols_calculations(self) -> dict:
        """calculates additional cols for the dataframes (emas, smas, macd, rsi)"""
        extra_cols_dataframe = {ticker: [] for ticker in self.subsampled_data.keys()}
        for ticker in self.subsampled_data.keys():
            dataframe = self.subsampled_data[ticker][0]
            # emas
            emas_used = [3, 5, 10, 12, 26, 30]
            smas_used = [10, 30, 50]
            for x in emas_used:
                ema = x
                dataframe["ema_" + str(ema)] = round(
                    dataframe.iloc[:, 0].ewm(span=ema, adjust=False).mean(), 2
                )
            # smas
            for x in smas_used:
                sma = x
                dataframe["sma_" + str(sma)] = round(
                    dataframe.iloc[:, 0].ewm(span=sma, adjust=False).mean(), 2
                )
            # macd
            dataframe["macd"] = dataframe["ema_12"] - dataframe["ema_26"]
            # rsi
            up_prices, down_prices = rsi_calc.up_down(dataframe["close"].values)
            avg_gain, avg_loss = rsi_calc.averages(up_prices, down_prices)
            RS, RSI = rsi_calc.rsi(dataframe["close"].values, avg_gain, avg_loss)
            dataframe["rsi"] = RSI
            # social media sentiment

            ## add the new dataframe to extra_cols_dataframe
            dataframe = dataframe[
                [
                    "close",
                    "open",
                    "high",
                    "low",
                    "volume",
                    "ema_3",
                    "ema_5",
                    "ema_10",
                    "ema_30",
                    "sma_10",
                    "sma_30",
                    "sma_50",
                    "macd",
                    "rsi",
                ]
            ]
            extra_cols_dataframe[ticker] = [dataframe, self.subsampled_data[ticker][1]]
        return extra_cols_dataframe

    def create_future_dates(self):
        date_time = self.subsampled_data[self.symbols[0]][
            1
        ]  # this is assuming all tickers trade on the same day
        next_dates = self.next_trading_days()

        total_dates = (
            date_time.tolist() if not isinstance(date_time, list) else date_time
        )
        for date in next_dates:
            total_dates.append(date)
        return total_dates, next_dates

    def next_trading_days(self) -> list:
        nyse = mcal.get_calendar("NYSE")
        new_dates = []
        # grab 6 months worth of future trading days to then find the needed dates within this list
        self.last_date = pd.to_datetime(self.last_date)
        trading_days = nyse.valid_days(
            start_date=self.last_date + datetime.timedelta(days=1),
            end_date=self.last_date + datetime.timedelta(days=6 * 30),
        )
        if self.timescale == "days":
            for date in trading_days:
                new_dates.append(date.strftime("%Y-%m-%d %H:%M:%S"))
            new_dates = new_dates[: self.future]
            return new_dates
        elif self.timescale == "mins":
            # checking to see if we are predicting less than a trading day ahead
            if self.future <= 390:
                schedule = nyse.schedule(
                    start_date=trading_days[0], end_date=trading_days[0]
                )
                diff = (
                    390 - self.future
                )  # how far off our prediction is to the end of the actual day
            else:
                # if more than a full day then get self.future in num of days and round up
                schedule = nyse.schedule(
                    start_date=trading_days[0],
                    end_date=trading_days[int(self.future / 390)],
                )
                diff = (
                    1 + int(self.future / 390)
                ) * 390 - self.future  # how far off our prediction is to the end of the actual day
            # create a list of datetime objects for the schedule times
            schedule.index = pd.to_datetime(schedule.index)
            for date in list(schedule.index.values):
                if date == list(schedule.index.values)[-1]:
                    mins_difference = 390 - diff
                else:
                    mins_difference = 390
                for i in range(mins_difference):
                    transformed_date = schedule.loc[date][
                        "market_open"
                    ] + datetime.timedelta(minutes=i)
                    transformed_date = pd.to_datetime(transformed_date)
                    new_dates.append((transformed_date.strftime("%Y-%m-%d %H:%M:%S")))
            return new_dates


def model_exists(path_vars) -> bool:
    symbol, shift, timescale, model_name = (
        path_vars[0],
        path_vars[1],
        path_vars[2],
        path_vars[3],
    )
    model_path = os.path.join(
        _models, model_name
    )  # currently checking if multi variable models exist
    return os.path.exists(model_path)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True) -> pd.DataFrame:
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
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
    x_test = x_test.reshape((x_test.shape[0], batch * num_features))

    x_test_and_pred = np.concatenate((yhat, x_test[:, 1 - num_features :]), axis=1)
    inv_yhat = scaler.inverse_transform(x_test_and_pred)

    y_test = y_test.reshape((len(y_test), 1))
    inv_y = np.concatenate((y_test, x_test[:, 1 - num_features :]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)

    rmse = np.sqrt(mean_squared_error(inv_y[:, 0], inv_yhat[:, 0]))
    predicted_data = pd.DataFrame({name: [] for name in var.column_names})
    actual_data = pd.DataFrame({name: [] for name in var.column_names})

    valid_plot = pd.DataFrame({"time": [], "close": [], "predicted": []})
    shift = {"AAPL": batch + 271, "TSLA": batch + 945, "GME": batch + 5460}
    shift = batch if ML.timescale == "mins" else batch
    predicted_data["time"], actual_data["time"] = date_time.tolist(), date_time.tolist()

    predicted_data = predicted_data[
        -inv_yhat.shape[0] :
    ]  # fix some issues with predicted and invyhat misshape]
    actual_data = actual_data[
        -inv_yhat.shape[0] :
    ]  # fix some issues with predicted and invyhat misshape]

    # print(predicted_data.shape, inv_yhat.shape)
    i = 0
    for name in var.column_names:
        predicted_data[name] = inv_yhat[:, i]
        actual_data[name] = inv_y[:, i]
        i += 1

    predicted_data = predicted_data.drop_duplicates(subset=["time"])
    actual_data = actual_data.drop_duplicates(subset=["time"])

    valid_plot["time"], valid_plot["close"], valid_plot["predicted"] = (
        pd.to_datetime(predicted_data["time"]),
        actual_data["close"],
        predicted_data["close"],
    )

    predicted_data.set_index("time", inplace=True)
    actual_data.set_index("time", inplace=True)

    # for the min timescale, reduce the scope of the plot for better granularity and to remove timescale as the plotting index
    valid_plot_plot = valid_plot.copy()
    if var.timescale == "mins":
        valid_plot_plot["time"] = valid_plot_plot["time"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        valid_plot_plot = valid_plot_plot[
            -60 * 16 * 2 :
        ]  # grabs last 2 days of min data
    else:
        valid_plot_plot["time"] = valid_plot_plot["time"].dt.strftime("%Y-%m-%d")

    valid_plot_plot.set_index("time", inplace=True)
    valid_plot.set_index("time", inplace=True)

    # plot
    fig2, ax2 = plt.subplots()
    ax2.set_title(
        f"{symbol} Predicted Prices for {var.timescale} from historical data: "
    )
    ax2.set_xlabel(f"Dates")
    ax2.set_ylabel(f"Price (USD)")
    splits = (
        int(valid_plot_plot.shape[0] / 100) if valid_plot_plot.shape[0] > 100 else 1
    )
    ax2.plot(valid_plot_plot[["close"]][::splits])
    ax2.plot(valid_plot_plot[["predicted"]][::splits], marker="x")
    ax2.legend(["Actual", "Predicted"], loc="lower right")
    plt.xticks(rotation=90)

    # plt.show()
    return predicted_data, actual_data, valid_plot


@tf.function
def prediction_loop(input_data, input_objs, input_vars):
    x_total, predicted_data, df = input_data
    scaler, model = input_objs
    batch, num_features = input_vars

    test_data = x_total[-1:]  # latest data point
    yhat = model.predict(test_data)

    test_data = test_data.reshape((test_data.shape[0], batch * num_features))
    test_data = np.concatenate((yhat, test_data[:, 1 - num_features :]), axis=1)
    # print("shape for inverse transform \n", test_data.shape)

    i = 0
    for name in var.column_names:
        predicted_data[name] = scaler.inverse_transform(test_data)[:, i]
        i += 1

    # predicted_data['close'], predicted_data['open'], predicted_data['high'], predicted_data['low'], predicted_data['volume'] = scaler.inverse_transform(test_data)[:,0], scaler.inverse_transform(test_data)[:,1], scaler.inverse_transform(test_data)[:,2], scaler.inverse_transform(test_data)[:,3], scaler.inverse_transform(test_data)[:,4]

    # print("\n predicted df \n", predicted_data)
    df = pd.concat([df, predicted_data])
    try:
        df = df.set_index(["time"])
    except KeyError:
        pass

    return predicted_data, df


def transform_data(dataframe, n, num_features):
    # redefine class variables for the function
    date_time = var.subsampled_data[symbol][1]
    batch = ML.batch

    # transform and scale dataframe values
    values = dataframe.values
    values = values.astype("float32")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)

    total_df = series_to_supervised(data=scaled_values, n_in=batch)

    # split the data into train and test values
    total_values = total_df.values
    # previous definition of n is based on before series_to_supervised transformation
    n = total_values.shape[0]
    train_values = total_values[: int(n * 0.7), :]
    test_values = total_values[int(n * 0.7) :, :]
    print(total_values.shape, train_values.shape, test_values.shape)

    # split the train and test values into x and y (x is the input and y is the expected output)
    x_train, y_train = (
        train_values[:, : batch * num_features],
        train_values[:, -num_features],
    )
    x_train = x_train.reshape((x_train.shape[0], batch, num_features))

    x_test, y_test = (
        test_values[:, : batch * num_features],
        test_values[:, -num_features],
    )
    x_test = x_test.reshape((x_test.shape[0], batch, num_features))

    x_total, y_total = (
        total_values[:, : batch * num_features],
        total_values[:, -num_features],
    )
    x_total = x_total.reshape((x_total.shape[0], batch, num_features))
    return x_train, y_train, x_test, y_test, x_total, y_total, scaler


def create_and_predict(symbol):
    ## setting all the variables
    symbol = symbol

    df = (
        var.extra_cols_data[symbol][0]
        if ML.extra_cols_bool
        else var.subsampled_data[symbol][0]
    )
    date_time = (
        var.extra_cols_data[symbol][1]
        if ML.extra_cols_bool
        else var.subsampled_data[symbol][1]
    )
    string_date = str(dates.today).replace("-", "")
    model_name = (
        f"{string_date}-{symbol}-{ML.timescale}-{ML.epochs}epochs-extracol"
        if ML.extra_cols_bool
        else f"{symbol}-{ML.timescale}-{ML.epochs}epochs"
    )
    valid_file_name = (
        f"{symbol}-{ML.timescale}-{ML.epochs}epochs-extracol-validation.csv"
        if var.extra_cols_bool
        else f"{symbol}-{ML.timescale}-{ML.epochs}epochs-validation.csv"
    )

    timescale = ML.timescale
    epochs = ML.epochs
    validate, train_model = train_bool, train_bool
    batch = ML.batch

    n = df.shape[0]
    num_features = df.shape[1]
    last_date = date_time.iloc[-1]

    total_dates = var.total_dates
    next_dates = var.next_dates

    ## transform our data and get data for training and plotting
    x_train, y_train, x_test, y_test, x_total, y_total, scaler = transform_data(
        dataframe=df, n=n, num_features=num_features
    )

    ## define and train the model
    if train_model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(
                    units=50,
                    return_sequences=False,
                    input_shape=(x_train.shape[1], x_train.shape[2]),
                ),
                tf.keras.layers.Dense(units=1),
            ]
        )
        # train the model
        model.compile(optimizer="adam", loss="mean_squared_error")
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch,
            validation_data=(x_test, y_test),
            verbose=1,
            shuffle=True,
        )
        # save the model
        model.save(os.path.join(_models, model_name))
        # quick plot of train losses vs test losses
        fig1, ax1 = plt.subplots()
        ax1.plot(history.history["loss"], label="train")
        ax1.plot(history.history["val_loss"], label="test")
        ax1.set_xlabel(f"Number of Epochs")
        ax1.set_ylabel(f"Loss Value")
        ax1.set_title(f"{symbol} train loss vs test loss: ")

    ## verify model performance compared to actual data
    model = tf.keras.models.load_model(os.path.join(_models, model_name))
    if validate:
        print("Verifying Models Predictions...")
        input_data = [x_test, date_time, y_test]
        input_objs = [model, scaler]
        input_vars = [symbol, batch, num_features, n]
        predicted_data, actual_data, valid_plot = verify_model_predictions(
            input_data, input_objs, input_vars
        )

        # add to validation csv (to compare between models)
        valid_plot.to_csv(os.path.join(var._dirs[symbol][2], valid_file_name))
        validation_csv.run(valid_file_name, symbol)

        ## writes up historical accuracy of model
        comparison_csv.run(valid_file_name, symbol)

    ## make new predictions
    if predict_future:
        print("Predicting New Close Prices...")
        predicted_data = pd.DataFrame({name: [] for name in var.column_names})

        actual_df = df.copy()
        start = time.time()
        for i in range(ML.future):
            if i != 0:
                total_values = series_to_supervised(
                    data=scaler.fit_transform(df.values), n_in=batch
                ).values
                x_total_test = total_values[:, : batch * num_features]
                x_total = x_total_test.reshape(
                    (x_total_test.shape[0], batch, num_features)
                )[-1:]
            input_data = [x_total, predicted_data, df]
            input_objs = [scaler, model]
            input_vars = [batch, num_features]
            prediction_data, df = prediction_loop(input_data, input_objs, input_vars)
            print(f"prediction for {ML.timescale} {i} done!")
        print(f"Time to make predictions: {time.time() - start}s")

        df["time"] = pd.to_datetime(total_dates)
        df = df.drop_duplicates(subset=["time"])
        df = df.set_index(["time"])

        actual_df["time"] = pd.to_datetime(total_dates[: -ML.future])
        actual_df = actual_df.drop_duplicates(subset=["time"])
        actual_df = actual_df.set_index(["time"])

        fig3, ax3 = plt.subplots()
        ax3.set_title(
            f"{symbol}s Historical Data and {ML.future} New Predicted Prices: "
        )
        ax3.set_xlabel(f"Dates")
        ax3.set_ylabel(f"Price (USD)")
        splits = int(df.shape[0] / 100) if df.shape[0] > 100 else 1
        ax3.plot(df["close"][int(0.5 * df.shape[0]) :: splits], marker="x")
        ax3.plot(actual_df["close"][int(0.5 * df.shape[0]) :: splits])
        ax3.legend(["predicted", "actual"], loc="lower right")

        df.to_csv(os.path.join(var._dirs[symbol][1], valid_file_name))

        print(
            f"\n prediction data (last {5 + ML.future} values): \n {df.tail(5 + ML.future)}"
        )
        print(f"\n actual data (last 5 values): \n {actual_df.tail(5)}")


if __name__ == "__main__":
    # 'AAPL', 'TSLA', 'GME', 'ABNB', 'PLTR', 'ETSY', 'ENPH', 'GOOG', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'
    # 'BTC-USD', 'ETH-USD', 'NANO-USD', 'ADA-USD'
    symbols = [
        "AAPL",
        "TSLA",
        "GME",
        "ABNB",
        "PLTR",
        "ETSY",
        "ENPH",
        "GOOG",
        "AMZN",
        "IBM",
        "DIA",
        "IVV",
        "NIO",
    ]

    # symbols = [
    #     "AAPL",
    # ]

    ## runs the stock scraper if some previous days data is missing (scrapes minute data for past 14 days)
    scraper.run(symbols)

    dates = InitialiseDates()
    string_date = str(dates.today).replace("-", "")

    ML = InitialiseMLVars(future=1, timescale="mins")
    var = StockData(symbols=symbols, dates=dates, ML=ML)
    for symbol in var.symbols:
        """define training and predict bools based on if a model already exists"""
        model_name = (
            f"{string_date}-{symbol}-{ML.timescale}-{ML.epochs}epochs-extracol"
            if var.extra_cols_bool
            else f"{symbol}-{ML.timescale}-{ML.epochs}epochs"
        )
        train_bool = (
            False
            if model_exists(path_vars=[symbol, ML.future, ML.timescale, model_name])
            else True
        )
        predict_future = False if train_bool else True
        print(
            f"\n {symbol} \n {dates.today} \n Train / Validate: {train_bool} \n Predict: {predict_future} \n"
        )

        """ run the training/predicting script """
        create_and_predict(symbol=symbol)
        tf.keras.backend.clear_session()

    plt.show()
