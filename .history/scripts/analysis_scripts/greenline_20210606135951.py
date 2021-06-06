from datetime import datetime

import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf

def run(stocks, start_date):
    yf.pdr_override()
    today = datetime.now()

    for stock in stocks:
        df = pdr.get_data_yahoo(stock, start_date, today)

        df.drop(df[df['Volume'] < 1000].index, inplace=True)

        month_df = df.groupby(pd.Grouper(freq="M"))['High'].max()

        gl_date = 0
        last_glv = 0
        current_glv = 0
        current_date = ""


        for index, value in month_df.items():
            if value > current_glv:
                current_glv = value
                current_date = index
                counter = 0
            if value < current_glv:
                counter += 1

                if counter == 3 and ((index.month != today.month) or (index.year != today.year)):
                    if current_glv != last_glv:
                        # print(f"{stock} current GLV: {current_glv} on {index.strftime('%Y-%m-%d')}")
                        gl_date = current_date
                        last_glv = current_glv
                        counter = 0
        
        if last_glv == 0:
            message = f"{stock} has not formed a green line yet"
        else:
            message = f"{stock}'s Last Green Line: {last_glv} on {gl_date.strftime('%Y-%m-%d')}"
        print(message)

    
if __name__=='__main__':
    stocks = ['GME']
    start_date = datetime(1980, 12, 1)
    run(stocks, start_date)





