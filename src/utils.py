# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn import cross_validation

# Define the path of stock record file 
AAPL_PATH = "../data-set/AAPL.csv"
GOOG_PATH = "../data-set/GOOG.csv"
MSFT_PATH = "../data-set/MSFT.csv"


#######################################################################
# Apply z-score normalization
#
# Input:
#   dataframe:
#       Pandas DataFrame, could be any-dimension
#######################################################################
def applyZScore(dataframe):
    print(type(dataframe))
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
    cv = cross_validation.KFold(len(x), n_folds = k)

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


def main():
    pass

if __name__ == '__main__':
    main()