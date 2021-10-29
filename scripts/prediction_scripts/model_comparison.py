import os
import csv

import pandas as pd
import numpy as np
from numpy import nan as NaN
import datetime

import matplotlib.pyplot as plt


_root = os.getcwd()
_data = os.path.join(_root, "datasets\\validation")
_models = os.path.join(_root, "models")


def calculate_accuracy(df):
    change = []
    change.append(NaN)
    for n in range(len(df["predicted"]) - 1):
        if n == len(df["predicted"]) or n > len(df["predicted"]):
            print(n)
        change.append(
            100
            * (df["predicted"].iloc[n + 1] - df["close"].iloc[n])
            / df["close"].iloc[n]
        )
    return change


def run(comparison_file, ticker):
    path = os.path.join(_data, ticker)
    chosen_model = os.path.join(path, comparison_file)
    df = pd.read_csv(chosen_model)  # 0th value is the comparison table

    df["point_difference"] = df["predicted"] - df["close"]
    df["perc_close"] = df["close"].pct_change() * 100
    df["perc_predicted"] = df["predicted"].pct_change() * 100
    df["perc_difference"] = df["perc_predicted"] - df["perc_close"]
    df["predicted_error"] = calculate_accuracy(df)

    avg_point_difference = df["point_difference"].mean(axis=0)
    avg_perc_close = df["perc_close"].mean(axis=0)
    avg_perc_predicted = df["perc_predicted"].mean(axis=0)
    avg_perc_difference = df["perc_difference"].mean(axis=0)
    avg_error_predicted = df["predicted_error"].mean(axis=0)

    try:
        df2 = pd.read_csv(os.path.join(path, f"{ticker}-comparison-data.csv"))
    except FileNotFoundError:
        df2 = pd.DataFrame(
            columns=[
                "date",
                "model",
                "avg_point_difference",
                "avg_perc_close",
                "avg_perc_predicted",
                "avg_perc_difference",
                "avg_error_predicted",
            ]
        )

    df2 = df2.append(
        {
            "date": datetime.date.today(),
            "model": chosen_model.replace(_data + f"\\{ticker}\\", ""),
            "avg_point_difference": avg_point_difference,
            "avg_perc_close": avg_perc_close,
            "avg_perc_predicted": avg_perc_predicted,
            "avg_perc_difference": avg_perc_difference,
            "avg_error_predicted": avg_error_predicted,
        },
        ignore_index=True,
    )
    print(df2)

    new_path = os.path.join(path, f"{ticker}-comparison-data.csv")
    df2.to_csv(new_path)


if __name__ == "__main__":
    symbols = ["AAPL"]
    comparison_data = {}
    for ticker in symbols:
        path = os.path.join(_data, ticker)
        files = [os.path.join(path, f) for f in os.listdir(path) if ".csv" in f]
        comparison_data[ticker] = files

    comparison_file = comparison_data["AAPL"][1]

    run(comparison_file, ticker)
