import datetime as dt
from datetime import datetime

import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf

import matplotlib.pyplot as plt 
import matplotlib.dates as mdates



def pivots_plot(df, stock, days):
    pivots = []
    dates = []
    dates_string = []
    counter = 0
    last_pivot = 0

    ranges = [0 for i in range(10)]
    date_range = [0 for i in range(10)]

    for i in df.index:
        current_max = max(ranges, default=0)
        value = round(df['High'][i], 2)
    
        ranges = ranges[1:9]
        ranges.append(value)
        date_range = date_range[1:9]
        date_range.append(i)

        if current_max == max(ranges, default=0):
            counter += 1
        else:
            counter = 0

        if counter == 5:
            last_pivot = current_max
            dateloc = ranges.index(last_pivot)
            last_date = date_range[dateloc]

            pivots.append(last_pivot)
            dates_string.append(last_date.strftime('%Y-%m-%d'))
            dates.append(last_date)
    
    
    time_del = dt.timedelta(days=days)

    for index in range(len(pivots)):
        print(f'{pivots[index]} : {dates_string[index]}')
        plt.plot_date(
            [dates[index] - (time_del * 0.075), dates[index] + time_del],
            [pivots[index], pivots[index]],
            linestyle="--",
            linewidth=1,
            marker=','
        )
        plt.annotate(
            str(pivots[index]), 
            (mdates.date2num(dates[index]), pivots[index]),
            xytext=(-10, 7),
            textcoords='offset points',
            fontsize=7,
            arrowprops=dict(arrowstyle='-|>')
        )

    plt.title(f'{stock} Resistance and Pivot Points')
    plt.xlabel('Dates')
    plt.ylabel('Price (USD)')
    plt.ylim(df['Low'].min(), df['High'].max() * 1.05)

def run(stocks, start_date):
    yf.pdr_override()
    today = datetime.now() 

    for stock in stocks:
        df = pdr.get_data_yahoo(stock, start_date, today)

        df['High'].plot(label='high')

        pivots_plot(df=df, stock=stock, days=30)
        
        plt.show()

if __name__=='__main__':
    stocks = ['BTC-USD']
    start_date = datetime(2021, 1, 1)
    run(stocks, start_date)

