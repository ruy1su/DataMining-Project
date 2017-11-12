# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn import cross_validation

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
# Input: np.ndarray in SAME dimension
#        [1, 1, 1, 1] vs. [1, 1, 0, 1]
#     or [[1],[1],[1],[1]] vs. [[1],[1],[0],[1]]
def computeMSE(predicted_y, test_y):
    mse = np.sum((predicted_y - test_y) ** 2) / predicted_y.shape[0]
    return mse

def computeRecall(predicted_y, test_y):
    TP, FN = 0, 0

    for i in range(len(predicted_y)):
        if((predicted_y[i] == 1) and (test_y[i] == 1)):
            TP += 1
        elif((predicted_y[i] == 0) and (test_y[i] == 1)):
            FN += 1

    ret = TP / (TP + FN) if (TP + FN) != 0 else 0

    return ret

def computePrecision(predicted_y, test_y):
    TP, FP = 0, 0

    for i in range(len(predicted_y)):
        if((predicted_y[i] == 1) and (test_y[i] == 1)):
            TP += 1
        elif((predicted_y[i] == 1) and (test_y[i] == 0)):
            FP += 1

    ret = TP / (TP + FP) if (TP + FP) != 0 else 0

    return ret

def computeAccuracy(predicted_y, test_y):
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(predicted_y)):
        if((predicted_y[i] == 1) and (test_y[i] == 1)):
            TP += 1
        elif((predicted_y[i] == 1) and (test_y[i] == 0)):
            FP += 1
        elif((predicted_y[i] == 0) and (test_y[i] == 1)):
            FN += 1
        else:
            TN += 1

    ret = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0

    return ret

# sign function, convert continuous value into two classes [0, 1]
# Input vec is a DataFrame
# Return a new DataFrame with each item applied sign function
def sign(vec):
    # value of vec will not be changed
    new_vec = vec.applymap(lambda x: 1 if x >= 0 else 0)  

    return new_vec

# Perform K-Fold testing
def KfoldTester(model, x, y, k):
    cv = cross_validation.KFold(len(x), n_folds = k)

    

    for train_idx, test_idx in cv:
        train_x = x.values[train_idx]
        train_y = y.values.ravel()[train_idx]

        test_x = x.values[test_idx]
        test_y = y.values.ravel()[test_idx]

        train_x = {'x'+str(i):[train_x[j][i] for j in range(len(train_x))] for i in range(len(train_x[0]))}
        train_x = pd.DataFrame(train_x)

        train_y = {'y':train_y}
        train_y = pd.DataFrame(train_y)

        test_x = {'x'+str(i):[test_x[j][i] for j in range(len(test_x))] for i in range(len(test_x[0]))}
        test_x = pd.DataFrame(test_x)

        test_y = {'y':test_y}
        test_y = pd.DataFrame(test_y)

        model.train(train_x, train_y)
        model.test(test_x, test_y)


def main():
    pass

if __name__ == '__main__':
    main()