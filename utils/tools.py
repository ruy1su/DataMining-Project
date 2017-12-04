# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn import cross_validation
import matplotlib.pyplot as plt

# Define the path of stock record file 
AAPL_PATH = "../data-set/stocks/AAPL.csv"
GOOG_PATH = "../data-set/stocks/GOOG.csv"
MSFT_PATH = "../data-set/stocks/MSFT.csv"

# Define the path of sentimental analysis result
# Negative / Positive
AAPL_MOOD_PN = "../data-set/sentiments/AAPL-mood-pn.csv"
GOOG_MOOD_PN = "../data-set/sentiments/GOOG-mood-pn.csv"
MSFT_MOOD_PN = "../data-set/sentiments/MSFT-mood-pn.csv"
# Multi-dimension
AAPL_MOOD_MULTI = "../data-set/sentiments/AAPL-mood-multi.csv"
GOOG_MOOD_MULTI = "../data-set/sentiments/GOOG-mood-multi.csv"
MSFT_MOOD_MULTI = "../data-set/sentiments/MSFT-mood-multi.csv"


#######################################################################
# Apply z-score normalization
#
# Input:
#   dataframe:
#       Pandas DataFrame, could be any-dimension
#######################################################################
def applyZScore(dataframe):
    expectation = np.mean(dataframe, axis=0)
    std_dev = np.std(dataframe, axis=0)

    normalized_dataframe = (dataframe - expectation) / std_dev

    return normalized_dataframe


#######################################################################
# Compute Mean Square Error (MSE)
#
# Input:
#   predicted_y:
#       predicted label y
#       numpy ndarray (n dimension or n * 1 dimension)
#   test_y:
#       ground truth label y
#       numpy ndarray (n dimension or n * 1 dimension)
# 
# Output:
#   MSE: Mean Squre Error
#
# Note:
#   predicted_y and test_y should be in the SAME dimension
#   e.g., [1, 1, 1, 1] vs. [1, 1, 0, 1] 
#      or [[1],[1],[1],[1]] vs. [[1],[1],[0],[1]]
#######################################################################
def computeMSE(predicted_y, test_y):
    mse = np.sum((predicted_y - test_y) ** 2) / predicted_y.shape[0]
    return mse


#######################################################################
# Compute Recall - TP / (TP + FN)
#
# Input:
#   predicted_y:
#       predicted label vector
#       type: numpy array (n dimension)
#       [y1, y2, y3, ..., yn]
#   test_y:
#       ground truth label vector
#       type: numpy array (n dimension)
#       [y1, y2, y3, ..., yn]
#######################################################################
def computeRecall(predicted_y, test_y):
    TP, FN = 0, 0

    for i in range(len(predicted_y)):
        if((predicted_y[i] == 1) and (test_y[i] == 1)):
            TP += 1
        elif((predicted_y[i] == 0) and (test_y[i] == 1)):
            FN += 1

    ret = TP / (TP + FN) if (TP + FN) != 0 else 0

    return ret


#######################################################################
# Compute Precision - TP / (TP + FP)
#
# Input:
#   predicted_y:
#       predicted label vector
#       type: numpy array (n dimension)
#       [y1, y2, y3, ..., yn]
#   test_y:
#       ground truth label vector
#       type: numpy array (n dimension)
#       [y1, y2, y3, ..., yn]
#######################################################################
def computePrecision(predicted_y, test_y):
    TP, FP = 0, 0

    for i in range(len(predicted_y)):
        if((predicted_y[i] == 1) and (test_y[i] == 1)):
            TP += 1
        elif((predicted_y[i] == 1) and (test_y[i] == 0)):
            FP += 1

    ret = TP / (TP + FP) if (TP + FP) != 0 else 0

    return ret


#######################################################################
# Compute Recall - (TP + TN) / ALL
#
# Input:
#   predicted_y:
#       predicted label vector
#       type: numpy array (n dimension)
#       [y1, y2, y3, ..., yn]
#   test_y:
#       ground truth label vector
#       type: numpy array (n dimension)
#       [y1, y2, y3, ..., yn]
#######################################################################
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


#######################################################################
# sign function
#   y = 1 if x >= 0, otherwise y = 0
# 
# Input:
#   vec: Pandas DataFrame in any dimension (Cannot be Series)
#######################################################################
def sign(vec):
    # value of vec will not be changed
    new_vec = vec.applymap(lambda x: 1 if x >= 0 else 0)  

    return new_vec

def sign_list(arr):
    return list(map(lambda x: 1 if x >= 0 else 0, arr))


#######################################################################
# K-Fold Dataset Generator
#   This function will generate training set and testing set for
#   k-fold cross validation
#
# Input:
#   x:  
#       feature vector data set
#       type: Pandas DataFrame (n * p dimension)
#       x.values: [[x11, x12, ..., x1p], ..., [xn1, ..., xnp]]
#   y:  
#       label vector data set
#       type: Pandas DataFrame (n * 1 dimension)
#       y.values: [[y1], [y2], [y3], ..., [yn]]
#   k:
#       K-Fold parameter (how any fold?)
#
# Output: A zip of (ret_train_x, ret_train_y, ret_test_x, ret_test_y)
#   ret_train_x: training feature vectors set
#       type: Pandas DataFrame (n * p dimension)
#       x.values: [[x11, x12, ..., x1p], ..., [xn1, ..., xnp]]
#   ret_train_y: training label vectors set
#       type: Pandas DataFrame (n * 1 dimension)
#       y.values: [[y1], [y2], [y3], ..., [yn]]
#   ret_test_x: testing feature vectors set
#       type: Pandas DataFrame (n * p dimension)
#       x.values: [[x11, x12, ..., x1p], ..., [xn1, ..., xnp]]
#   ret_test_y: testing label vectors set
#       type: Pandas DataFrame (n * 1 dimension)
#       y.values: [[y1], [y2], [y3], ..., [yn]]
#######################################################################
def KfoldGenerator(x, y, k):
    cv = cross_validation.KFold(len(x), n_folds = k, shuffle=False, random_state=None)

    ret_train_x, ret_train_y = [], []
    ret_test_x, ret_test_y = [], []

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

        ret_train_x.append(train_x)
        ret_train_y.append(train_y)
        ret_test_x.append(test_x)
        ret_test_y.append(test_y)

    return zip(ret_train_x, ret_train_y, ret_test_x, ret_test_y)


#######################################################################
# Get next day
# For example, next day of 2010-12-31 is 2011-01-01
# 
# Input:
#   date: current day, in the format of year-month-day
#   year: specify year if date == ""
#   month: specify month if date == ""
#   day: specify day if date == ""
# 
# Output:
#   next day of input day, in the format of year-month-day
#
# Usage: nextDay("2010-01-01") or nextDay(year=2011, month=12, day=31)
#######################################################################
def nextDay(date="", year=0, month=0, day=0):
    if(date != ""):
        date = date.split('-')
        year = eval(date[0])
        month = eval(date[1] if date[1][0] != '0' else date[1][1])
        day = eval(date[2] if date[2][0] != '0' else date[2][1])

    month_days = [[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                  [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]]

    # if is_leap == 0, indicates current year is not leap year
    is_leap = 0 if (year % 4 != 0 or (year % 100 == 0 and year % 400 != 0)) else 1

    # update day, month, year
    day = 1 if (month_days[is_leap][month - 1] == day) else day + 1
    month = 1 if (day == 1 and month == 12) else (month + 1 if (day == 1) else month)
    year = year + 1 if (month == 1 and day == 1) else year

    return ("%04d-%02d-%02d" % (year, month, day))

def plot(predict, real, num=50, print=False, classification=False):
        if num > len(predict):
            num = len(predict)
        if print:
            print(predict[:num])
            print(real[:num])
        if classification:
            predict = plt.plot(utils.sign_list(predict[:num]), 'bo', label='predict')
            real = plt.plot(utils.sign_list(real[:num]), 'ro', label='real')
        else:
            plt.plot(predict[:num], 'b', label='predict')
            plt.plot(real[:num], 'r', label='real')

        plt.xlabel('day')
        plt.ylabel('fluctuation')
        plt.legend(loc='best')
        plt.show()


def main():
    pass


if __name__ == '__main__':
    main()