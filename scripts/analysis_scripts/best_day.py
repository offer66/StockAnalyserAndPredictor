import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError
import yfinance as yf

import matplotlib.pyplot as plt

_root = os.getcwd()
_portfolio = os.path.join(_root, 'datasets\\portfolio\\')
_plots = os.path.join(_portfolio, 'plots')

def run():
    pass

if __name__=="__main__":
    run()
    plt.show()