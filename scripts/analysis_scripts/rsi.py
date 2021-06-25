from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError
import yfinance as yf 

import matplotlib.pyplot as plt



def up_down(close_prices):
    up_prices = []
    down_prices = []
    for i in range(len(close_prices)):
        if i == 0:
            up_prices.append(0)
            down_prices.append(0)
        else:
            if close_prices[i] - close_prices[i-1] > 0:
                up_prices.append(close_prices[i] - close_prices[i-1])
                down_prices.append(0)
            else:
                down_prices.append(close_prices[i] - close_prices[i-1])
                up_prices.append(0)
    return up_prices, down_prices


def averages(up_prices, down_prices):
    avg_gain = []
    avg_loss = []
    x = 0
    while x < len(up_prices):
        if x < 15:
            avg_gain.append(0)
            avg_loss.append(0)
        else:
            sum_gain = 0
            sum_loss = 0
            y = x - 14
            while y<=x:
                sum_gain += up_prices[y]
                sum_loss += down_prices[y]
                y += 1
            avg_gain.append(sum_gain/14)
            avg_loss.append(abs(sum_loss/14))
        x += 1
    return avg_gain, avg_loss


def rsi(close_prices, avg_gain, avg_loss):
    RS = []
    RSI = []
    for p in range(len(close_prices)):
        if p < 15 or avg_loss[p] == 0:
            RS.append(0)
            RSI.append(0)
        else:
            RSvalue = (avg_gain[p] / avg_loss[p])
            RS.append(RSvalue)
            RSI.append(100 - (100 / (1 + RSvalue)))
    return RS, RSI


def compare_days(stock, close_prices, RSI):
    # variables = {
    #     'Days_Observed': 15, 'Crosses': 0, 'Nothing': 0, 
    #     'True_Positive': 0, 'False_Positive': 0, 'True_Negative': 0, 'False_Negative': 0, 
    #     'Sensitivity': 0, 'Specificity': 0, 'Accuracy': 0
    # }
    Days_Observed = 15
    Crosses = 0
    nothing = 0
    True_Positive = 0
    False_Positive = 0
    True_Negative = 0
    False_Negative = 0
    Sensitivity = 0
    Specificity = 0
    Accuracy = 0
    while Days_Observed < len(close_prices)-5:
        if RSI[Days_Observed] <= 30:
            if ((close_prices[Days_Observed + 1] + close_prices[Days_Observed + 2] + close_prices[Days_Observed + 3] + close_prices[Days_Observed + 4] + close_prices[Days_Observed + 5])/5) > close_prices[Days_Observed]:
                True_Positive += 1
            else:
                False_Negative += 1
            Crosses += 1
        elif RSI[Days_Observed] >= 70:
            if ((close_prices[Days_Observed + 1] + close_prices[Days_Observed + 2] + close_prices[Days_Observed + 3] + close_prices[Days_Observed + 4] + close_prices[Days_Observed + 5])/5) <= close_prices[Days_Observed]:
                True_Negative += 1
            else:
                False_Positive += 1
            Crosses += 1
        else:
            nothing+=1
        Days_Observed += 1
    try:
        Sensitivity = (True_Positive / (True_Positive + False_Negative))
    except ZeroDivisionError:
        Sensitivity = 0
    try:
        Specificity = (True_Negative / (True_Negative + False_Positive))
    except ZeroDivisionError:
        Specificity = 0
    try:
        Accuracy = (True_Positive + True_Negative) / (True_Negative + True_Positive + False_Positive + False_Negative)
    except ZeroDivisionError:
        Accuracy = 0

    TPR = Sensitivity
    FPR = 1 - Specificity
    # Create a row to add to the compare_stocks
    add_row = {'Company' : stock, 'Current_RSI': RSI[-1], 'Days_Observed' : Days_Observed, 'Crosses' : Crosses, 'True_Positive' : True_Positive, 'False_Positive' : False_Positive, 
    'True_Negative' : True_Negative, 'False_Negative' : False_Negative, 'Sensitivity' : Sensitivity, 'Specificity' : Specificity, 'Accuracy' : Accuracy, 'TPR' : TPR, 'FPR' : FPR}

    return add_row


def run(stock, start_date, rsi_comparison):
    # yf.pdr_override()
    today = dt.today().strftime('%Y-%m-%d')

    try:
        df = pdr.DataReader(f'{stock}-USD','yahoo', start_date, today)
    except RemoteDataError:
        df = pdr.get_data_yahoo(stock, start_date, today)

    close_prices = df['Close'].values

    # do rsi caluclations
    up_prices, down_prices = up_down(close_prices)
    avg_gain, avg_loss = averages(up_prices, down_prices)
    RS, RSI = rsi(close_prices, avg_gain, avg_loss)

    # put rsi calculations into a dict
    df_dict = {
        'Prices' : close_prices,
        'upPrices' : up_prices,
        'downPrices' : down_prices,
        'AvgGain' : avg_gain,
        'AvgLoss' : avg_loss,
        'RS' : RS,
        'RSI' : RSI
    }
    df_rsi = pd.DataFrame(df_dict, columns = ['close_prices', 'up_prices', 'down_prices', 'avg_gain','avg_loss', 'RS', "RSI"])

    # get price predictions from rsi calculations
    add_row = compare_days(stock, close_prices, RSI)
    rsi_comparison = rsi_comparison.append(add_row, ignore_index = True) # Add the analysis on the stock to the existing Compare_Stocks dataframe
    
    return rsi_comparison, np.mean(avg_gain), np.mean(avg_loss)


if __name__ == '__main__':
    symbols = ['BTC-USD', 'ETH-USD']
    start_date = '2019-01-01'
    rsi_comparison = pd.DataFrame(columns=[
        "Company", "Current_RSI", "Days_Observed", "Crosses", 
        "True_Positive", "False_Positive", "True_Negative", "False_Negative", 
        "Sensitivity", "Specificity", "Accuracy", "TPR", "FPR"]
    )
    print("Note that 0% - 30% RSI is undervalued and 70% - 100% RSI is overvalued!")
    for ticker in symbols:
        rsi_comparison, avg_gain, avg_loss = run(stock=ticker, start_date=start_date, rsi_comparison=rsi_comparison)
        print('---------------------------------------------')
        print(f"Results for {ticker} going back to {start_date}")
        print(f"Average Gain: {avg_gain} %")
        print(f"Average Loss: {avg_loss} %")
        print(f"Total Profits: {avg_gain - avg_loss} %")
        print('---------------------------------------------')
    rsi_comparison = rsi_comparison.set_index('Company')
    print(rsi_comparison)

