import os
import sys
sys.path.append("scripts\\analysis_scripts\\")
sys.path.append("scripts\\prediction_scripts\\")

import pandas as pd


def run_portfolio_analysis():
    import basic_analysis as analysis
    ''' Run portfolio analysis '''
    analysis.run()


def run_scraper(symbols):
    import stock_scraper as scraper
    ''' Run stock scraper '''
    scraper.run(symbols)


def get_rsi(symbols):
    import rsi as rsi
    start_date = '2019-01-01'
    rsi_comparison = pd.DataFrame(columns=[
        "Company", "Current_RSI", "Days_Observed", "Crosses", 
        "True_Positive", "False_Positive", "True_Negative", "False_Negative", 
        "Sensitivity", "Specificity", "Accuracy", "TPR", "FPR"]
    )
    print("Note that 0% - 30% RSI is undervalued and 70% - 100% RSI is overvalued!")
    for ticker in symbols:
        rsi_comparison, avg_gain, avg_loss = rsi.run(stock=ticker, start_date=start_date, rsi_comparison=rsi_comparison, current=True)
    rsi_comparison = rsi_comparison.set_index('Company').sort_values(by='Current_RSI')
    print(rsi_comparison)



def run_ml(var, ML, dates):
    import model_create_predict as mcp
    ''' Run ML setup and prediction script '''
    dates = mcp.InitialiseDates()
    ML = mcp.InitialiseMLVars(future=1, timescale='mins', validate=True)
    var = mcp.StockData(symbols=symbols, dates=dates, ML=ML)

    
    for symbol in var.symbols:
        ''' define training and predict bools based on if a model already exists '''
        model_name = f'{symbol}-{ML.timescale}-{ML.epochs}epochs-extracol' if var.extra_cols_bool else f'{symbol}-{ML.timescale}-{ML.epochs}epochs'
        # train_bool = False if model_exists(path_vars=[symbol, ML.future, ML.timescale, model_name]) else True
        train_bool = False
        predict_future = False if train_bool else True
		
        ''' run the training/predicting script '''
        mcp.create_and_predict(symbol=symbol)


def run(symbols, features):
    if features["portfolio_analysis"]:
        print("Running portfolio analysis...")
        run_portfolio_analysis()

    elif features["scraper"]:
        print("Running stock scraper...")
        run_scraper(symbols)

    elif features["predictor"]:
        import model_create_predict as mcp
        print("Running ML predictor...")
        dates = mcp.InitialiseDates()
        ML = mcp.InitialiseMLVars(future=1, timescale='mins', validate=True)
        var = mcp.StockData(symbols=symbols, dates=dates, ML=ML)
        run_ml(var, ML, dates)

    elif features["rsi"]:
        print("Calculating most recent RSI value...")
        get_rsi(symbols)



if __name__=="__main__":
    # symbols = [
	# 	'GME', 'ABNB', 'PLTR', 'ETSY', 'ENPH', 'GOOG', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'
	# ]
    symbols = [
        'GME'
    ]
    features = {
        "portfolio_analysis": False,
        "scraper": False,
        "predictor": True,
        "rsi": False
    }
    run(symbols=symbols, features=features)
