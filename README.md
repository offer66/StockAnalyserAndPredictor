# Stock Analysis and Predictor

## Description

This project runs analysis on a specific set of stocks or a given portfolio; providing insight into portfolio diversity, backtest success using a variety of methods, and trading views.

Furthermore ML models can be trained to perform predictions into the future. Using a alpha-vantage API data pipeline that pulls minute by minute stock prices or yahoo finances' API for day by day prices, various models can be built to varying degrees of accuracy.

## Getting Started

### Installing

To look at the code just fork this repo and set up a virtual environment and install requirements.txt using
```
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r 'requirements.txt'
```

### Executing programs
#### Analysis
##### Portfolio Analysis
To run the scripts, edit data\portfolio_simple.csv to your own portfolio and run through the commands in basic_analysis.py.
```
python scripts/analysis_scripts/basic_analysis.py
```
This will produce an analysed portfolio_analysed.csv file with further information.
![alt text](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/images/analysed_csv.png?raw=true)


##### Strategy Backtesting
The backtesting strategy follow the red white blue strategy of buying and selling at different indicators in a stocks history. This can be used to determine if a stock has been previously successful using this specific strategy and to what degree of success it has had.

Edit the script so that the stocks list and start date are to your custom values and then run the script using
```
python scripts/analysis_scripts/backtest.py
```
![alt text](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/images/backtest.png?raw=true)

##### Greenline
This script calculates the last solid green line, providing the price and the date. These greenlines are good indicators for the last safe price of a stock.
```
python scripts/analysis_scripts/greenline.py
```
![alt text](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/images/greenline.png?raw=true)

##### Resistance and Pivots
This script calculates the pivot points of a stock and plots them. Once again the stocks and the start date can be edited within the script to customise your data.
```
python scripts/analysis_scripts/resistance_and_pivots.py
```
![alt text](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/images/pivots.png?raw=true)


##### Trading View
This script calculates shows a trading view of a stock. 
```
python scripts/analysis_scripts/trading_view.py
```
![alt text](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/images/tradingview.png?raw=true)


#### Predictions
##### Create and Run Models
This trains models based on a selection of stock symbols. Using a 'day' timescale the models can then predict into the future.

To use, edit the symbols list in model_create_predict.py

```
python scripts/prediction_scripts/model_create_predict.py
```

![alt text](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/images/predictor.png?raw=true)




## Authors

Contributors names and contact info

ex. Michael Cullen
michaelcullen2011@hotmail.co.uk


