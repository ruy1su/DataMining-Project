# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

AAPL_PATH = "../data-set/AAPL.csv"
GOOG_PATH = "../data-set/GOOG.csv"
MSFT_PATH = "../data-set/MSFT.csv"

def applyZScore(dataframe):
    expectation = np.mean(dataframe, axis=0)
    std_dev = np.std(dataframe, axis=0)

    normalized_dataframe = (dataframe - expectation) / std_dev

    return normalized_dataframe

# predicted_y and test_y are the predicted and actual y values respectively as numpy arrays
# function prints the mean squared error value for the test dataset
# Input: list
def compute_mse(predicted_y, test_y):
    mse = np.sum((predicted_y - test_y) ** 2) / predicted_y.shape[0]
    return mse

def main():
    pass

if __name__ == '__main__':
    main()