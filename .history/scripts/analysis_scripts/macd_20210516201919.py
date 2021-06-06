import pandas as pd
import numpy as np
import yfinance as yf 
from datetime import datetime as dt
from pandas_datareader import data as pdr
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

class DailyMACD(object):
    def __init__(
        self, ticker, start_date, short_prd, long_prd, signal_long_length,
        signal_short_length=0, tolerance=0.002, end_date=None, column="Close"
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.long = long_prd
        self.short = short_prd
        self.signal_long_length = signal_long_length
        self.purchase_prices = []
        self.sell_prices = []
        self.end_date = dt.today() if (end_date is None) else end_date
        self.data = None
        self.signal_short_length = signal_short_length
        self.column = column

        self.__get_data()

    def __get_data(self):
        try:
            self.data = pdr.DataReader(f'{self.ticker}-USD','yahoo', self.start_date, self.end_date)
            self.data = self.data.reset_index()
        except:
            self.data = pdr.get_data_yahoo(self.ticker, self.start_date, self.end_date)
            self.data = self.data.reset_index()

        self.data = self.data.sort_values("Date")
        self.ema_data  = self.data.loc[:(self.long + self.signal_long_length)-1]
        self.data = self.data.loc[(self.long + self.signal_long_length):]

        self.data_dates = pd.to_datetime(self.data.Date, format="%Y-%m-%d").tolist()
        self.ema_dates = pd.to_datetime(self.ema_data.Date, format="%Y-%m-%d").tolist()
        self.long_sma_data = self.ema_data.loc[:self.long-1][self.column]
        self.short_sma_data = self.ema_data.loc[:self.short-1][self.column]

    def __sma(self, N, price_hist):
        return sum(price_hist) / N

    def __ema(self, N, curr_price, past_ema):
        # "Smoothing Factor"
        k = 2 / (N + 1)
        ema = (curr_price * k) + (past_ema * (1-k))
        return ema

    def get_macd(self):
        return self.macd

    def get_signal(self):
        return self.signal

    def get_long_ema(self):
        return self.long_ema

    def get_short_ema(self):
        return self.short_ema

    def get_buy_sell_dates(self):
        buy_dates = []
        sell_dates = []
        for i in self.buy_lines:
            try:
                buy_dates.append(self.data_dates[i])
            except IndexError:
                pass
        for i in self.sell_lines:
            try:
                sell_dates.append(self.data_dates[i])
            except IndexError:
                pass
        return buy_dates, sell_dates

    def purchase_prices(self):
        return self.sell_prices

    def sell_prices(self):
        pass

    def profit(self):
        return self.profit

    def get_data(self):
        return self.data

    def volatility(self):
        return np.std(self.data[self.column])

    def ticker_symbol(self):
        return self.ticker.upper()

    def get_buy_sell_profits(self):
        position = 0 # have we bought yet?
        self.purchase_prices = []
        self.sell_prices = []
        for i in range(len(self.data)):
            if i in self.buy_lines:
                if position == 0:   # must have sold before buying
                    position = 1
                    self.purchase_prices.append(self.data.iloc[i][self.column])
            if i in self.sell_lines:
                if position > 0:    # must purchase before selling
                    position = 0
                    self.sell_prices.append(self.data.iloc[i][self.column])
        if len(self.purchase_prices) > len(self.sell_prices):
            # purchased at the end, consider accumulated profit/loss
            self.sell_prices.append(self.data.iloc[-1][self.column])
        self.purchase_prices = np.asarray(self.purchase_prices)
        self.sell_prices = np.asarray(self.sell_prices)
        # as percentage/100
        self.profit = np.sum((self.sell_prices - self.purchase_prices)/self.purchase_prices)

        return self.purchase_prices, self.sell_prices, self.profit

    def view(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
        fig.suptitle(self.ticker.upper() + " - MACD (" + str(self.short) + ", " \
                    + str(self.long) + ", " + str(self.signal_long_length) + ")")

        ax2.set_title("MACD vs Signal")
        ax2.set_ylabel("EMA")
        ax2.set_xlabel("Date")
        # plot the macd and signal lines on the bottom
        ax2.plot(self.data_dates, self.macd, color="green", label="MACD")
        ax2.plot(self.data_dates, self.long_signal, color="red",
                        label=(str(self.signal_long_length) +"-Period EMA"))
        ax2.legend()
        ax2.grid(True) # looks a little nicer

        ax1.set_title("Price Data")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        # plot the long and short EMA lines
        ax1.plot(self.data_dates, self.long_ema, color="tomato", label="long ema")
        ax1.plot(self.data_dates, self.short_ema, color="olivedrab", label="short ema")
        # plot the price data
        ax1.plot(self.data_dates, self.data[self.column], color="blue", label="price")
        # display the buying and selling points
        for line in self.sell_lines:
            ax1.axvline(self.data_dates[line], color="red")
        for line in self.buy_lines:
            ax1.axvline(self.data_dates[line], color="green")
        ax1.legend()
        ax1.grid(True) # looks a little nicer

        # plt.show() # display the graph

    def reliability(self):
        Days_Observed = 0
        Crosses = 0
        True_Positive = 0
        False_Positive = 0
        True_Negative = 0
        False_Negative = 0
        Sensitivity = 0
        Specificity = 0
        Accuracy = 0
        # This list holds the closing prices of a stock
        prices_df = self.data
        # prices_df = df['Close']

        # Calculate exponentiall weighted moving averages:
        ema12 = prices_df.ewm(span=12).mean()
        ema26 = prices_df.ewm(span=26).mean()
        macd = []  # List to hold the MACD line values
        counter=0  # Loop to substantiate the MACD line
        while counter < (len(ema12)):
            macd.append(ema12.iloc[counter,0] - ema26.iloc[counter,0])  # Subtract the 26 day EW moving average from the 12 day.
            counter += 1
        macd_df = pd.DataFrame(macd)
        signal_df = macd_df.ewm(span=9).mean() # Create the signal line, which is a 9 day EW moving average
        signal = signal_df.values.tolist()  # Add the signal line values to a list.

        #  Loop to Compare the expected MACD crosses results to the actual results
        Day = 1
        prices = prices_df['Close'].values
        while Day < len(macd) - 5: # -1 to be able to use the last day for prediction, -5 so we can look at the 5 day post average.
            Prev_Day = Day-1
            # Avg_Closing_Next_Days = (prices[Day+1] + prices[Day+2] + prices[Day+3] + prices[Day+4] + prices[Day+5])/5 # To use 5 day average as a decision.
            Avg_Closing_Next_Days = (prices[Day+1] + prices[Day+2] + prices[Day+3])/3  # To use 3 day average as a decision.
            Days_Observed += 1  # Count how many days were observed
            if ((signal[Prev_Day] > macd[Prev_Day]) and (signal[Day] <= macd[Day])):  # when the signal line dips below the macd line (Expected increase over the next x days)
                Crosses += 1   # register that a cross occurred
                if (prices[Day] < Avg_Closing_Next_Days):  # Tests if the price increases over the next x days.
                    True_Positive += 1
                else:
                    False_Negative += 1

            if ((signal[Prev_Day] < macd[Prev_Day]) and (signal[Day] >= macd[Day])): # when the signal line moves above the macd line (Expected dip over the next x days)
                Crosses += 1
                if (prices[Day] > Avg_Closing_Next_Days):  # Tests if the price decreases over the next x days.
                    True_Negative += 1
                else:
                    False_Positive += 1
            Day += 1
        try:
            Sensitivity = (True_Positive / (True_Positive + False_Negative)) # Calculate sensitivity
        except ZeroDivisionError:  # Catch the divide by zero error
            Sensitivity = 0
        try:
            Specificity = (True_Negative / (True_Negative + False_Positive)) # Calculate specificity
        except ZeroDivisionError:
            Specificity
        try:
            Accuracy = (True_Positive + True_Negative) / (True_Negative + True_Positive + False_Positive + False_Negative) # Calculate accuracy
        except ZeroDivisionError:
            Accuracy = 0
        TPR = Sensitivity  # Calculate the true positive rate
        FPR = 1 - Specificity  # Calculate the false positive rate
        # Create a row to add to the compare_stocks
        add_row = {'Company' : self.ticker, 'Days_Observed' : Days_Observed, 'Crosses' : Crosses, 'True_Positive' : True_Positive, 'False_Positive' : False_Positive, 
        'True_Negative' : True_Negative, 'False_Negative' : False_Negative, 'Sensitivity' : Sensitivity, 'Specificity' : Specificity, 'Accuracy' : Accuracy, 'TPR' : TPR, 'FPR' : FPR} 
        return add_row

    def run(self):
        # use first <long/short> # of points to start the EMA 
        # since it depends on previous EMA
        long_sma_value = self.__sma(self.long, self.long_sma_data)
        short_sma_value = self.__sma(self.short, self.short_sma_data)
        self.long_ema = [long_sma_value]
        self.short_ema = [short_sma_value]

        # need to remove these values at the end
        # 'use up' the remainder of the data for the EMAs
        for index, v in self.ema_data[self.long:].iterrows():
            self.long_ema.append(self.__ema(self.long, v[self.column], self.long_ema[-1]))
        for index, v in self.ema_data[self.short:].iterrows():
            self.short_ema.append(self.__ema(self.short, v[self.column], self.short_ema[-1]))

        # calculate the EMA values for the long/short lines for the
        # actual data under consideration (non-EMA data)
        for index, value in self.data.iterrows():
            self.long_ema.append(self.__ema(self.long, value[self.column], self.long_ema[-1]))
            self.short_ema.append(self.__ema(self.short, value[self.column], self.short_ema[-1]))
        # remove the first few values from the short EMA list 
        # to catch up with the start of the long EMA list
        self.short_ema = self.short_ema[(self.long - self.short):]

        # create numpy arrays to easily difference EMAs
        self.long_ema = np.asarray(self.long_ema)
        self.short_ema = np.asarray(self.short_ema)
        self.macd = self.short_ema - self.long_ema

        # use the first N values to start signal line EMA calc
        signal_line_sma = self.__sma(self.signal_long_length, self.macd[-self.signal_long_length:])
        self.long_signal = [signal_line_sma]
        # calculate the signal line for the actual (non-EMA) data
        for m in self.macd[self.signal_long_length+1:]:
            self.long_signal.append(self.__ema(self.signal_long_length, m, self.long_signal[-1]))
        # remove first entry in signal since it was only used to start calc
        self.long_signal = self.long_signal[1:]
        # remove the first few values of macd/short/long 
        # emas to catch up with signal/data
        self.macd = self.macd[self.signal_long_length+1:]
        self.long_ema = self.long_ema[self.signal_long_length+1:]
        self.short_ema = self.short_ema[self.signal_long_length+1:]

        # get difference of MACD and signal to find crossings
        self.long_signal = np.asarray(self.long_signal)
        self.diffs = self.macd - self.long_signal

        self.buy_lines = []
        self.sell_lines = []
        for i in range(1, len(self.diffs)):
            # previous MACD was < signal and current is greater so  buy
            if self.diffs[i-1] < 0 and self.diffs[i] > 0:
                self.buy_lines.append(i)
            # previous MACD was > signal and current is less so  sell
            if self.diffs[i-1] > 0 and self.diffs[i] < 0:
                self.sell_lines.append(i)


if __name__ == '__main__':
    symbols = ['AAPL', 'TSLA', 'GME']
    start_date = '2019-01-01'
    macd = {}
    macd_comparison = pd.DataFrame(columns=[
        "Company", "Days_Observed", "Crosses", 
        "True_Positive", "False_Positive", "True_Negative", "False_Negative", 
        "Sensitivity", "Specificity", "Accuracy", "TPR", "FPR"]
    )
    for ticker in symbols:
        macd[ticker] = DailyMACD(ticker=ticker, start_date=start_date, short_prd=12, long_prd=26, signal_long_length=9)
        macd[ticker].run()
        buy, sell, profit = macd[ticker].get_buy_sell_profits()
        bd, sd = macd[ticker].get_buy_sell_dates()
        add_row = macd[ticker].reliability()
        macd_comparison = macd_comparison.append(add_row, ignore_index=True)
        print('---------------------------------------------')
        print("Buy Prices: \n", buy)
        print("Sell Prices: \n", sell)
        print(f"Total Profits: {np.round(profit * 100, 2)} %")
        print('---------------------------------------------')
        # print(macd[ticker].get_buy_sell_dates()[0], "\n", macd[ticker].get_buy_sell_dates()[1])
        # macd[ticker].view()
    print(f"MACD reliability for each ticker: \n {macd_comparison}")
    plt.show()

