import os
import csv

import pandas as pd
import numpy as np
from numpy import nan as NaN

import matplotlib.pyplot as plt


_root = os.getcwd()
_data = os.path.join(_root, 'datasets\\validation')
_models = os.path.join(_root, 'models')


def calculate_accuracy(df):
    change = []
    change.append(NaN)
    for n in range(len(df['predicted']) - 1):
        if n == len(df['predicted']) or n > len(df['predicted']):
            print(n)
        change.append(100 * (df['predicted'].iloc[n + 1] - df['close'].iloc[n]) / df['close'].iloc[n])
    return change


def run(valid_file, ticker):
    # validation_data = {}
    # files = [os.path.join(path, f) for f in os.listdir(path) if ".csv" in f]
    # validation_data[ticker] = files

    path = os.path.join(_data, ticker)
    chosen_model = os.path.join(path, valid_file)
    df = pd.read_csv(chosen_model)        # 0th value is the comparison table

    df['point_difference'] = df['predicted'] - df['close']
    df['perc_close'] = df['close'].pct_change() * 100
    df['perc_predicted'] = df['predicted'].pct_change() * 100
    df['perc_difference'] = df['perc_predicted'] - df['perc_close']
    df['predicted_error'] = calculate_accuracy(df)


    avg_point_difference = df['point_difference'].mean(axis=0)
    avg_perc_close = df['perc_close'].mean(axis=0)
    avg_perc_predicted = df['perc_predicted'].mean(axis=0)
    avg_perc_difference = df['perc_difference'].mean(axis=0)
    avg_error_predicted = df['predicted_error'].mean(axis=0)

    try:
        df2 = pd.read_csv(os.path.join(path, f'{ticker}-comparison-data.csv'))
    except FileNotFoundError:
        df2 = pd.DataFrame.from_dict({
            'data': ['avg_point_difference', 'avg_perc_close', 'avg_perc_predicted', 'avg_perc_difference', 'avg_error_predicted'],
        })

    chosen_model = chosen_model.replace(_data + f"\\{ticker}\\", "")
    df2[chosen_model] = [avg_point_difference, avg_perc_close, avg_perc_predicted, avg_perc_difference, avg_error_predicted]
    df2 = df2.set_index("data")

    # print(df)
    # print(df2)

    new_path = os.path.join(path, f'{ticker}-comparison-data.csv')

    df3 = df2.transpose()
    df3.to_csv(os.path.join(_data, f'{ticker}-comparison-data-transposed.csv'))
    df2.to_csv(new_path)

    # print("Average Point Difference: ", avg_point_difference)
    # print(f"Average Percent Change Close: {avg_perc_close} %")
    # print(f"Average Percent Change Predicted: {avg_perc_predicted} %")
    # print(f"How Variation Compares: {avg_perc_difference} %")
    print(f"Predicted Error from Close Results: {avg_error_predicted} %")


if __name__=="__main__":
    symbols = ['AAPL']
    validation_data = {}
    for ticker in symbols:
        path = os.path.join(_data, ticker)
        files = [os.path.join(path, f) for f in os.listdir(path) if ".csv" in f]
        validation_data[ticker] = files
    
    valid_file = validation_data['AAPL'][1]
    
    run(valid_file, ticker)


