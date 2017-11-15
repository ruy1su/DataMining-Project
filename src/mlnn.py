# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier

import utils
import stockParser as sp


class MLNN(object):
    #######################################################################
    # Constructor
    #
    # Input:
    #   hidden_layers:
    #       tuple, length = number of hidden layers
    #       The ith element represents the number of neurons 
    #       in the ith hidden layer.
    #   zscore:
    #       False: do NOT apply z-score normalization
    #       True: apply z-score normalization 
    #######################################################################
    def __init__(self, hidden_layers, zscore=False):
        self.hidden_layers_ = hidden_layers
        self.zscore_ = zscore

        self.model_ = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='logistic',
            hidden_layer_sizes=self.hidden_layers_, random_state=1)

    #######################################################################
    # Model Training function
    # Input:
    #   x:  
    #       feature vector data set
    #       type: Pandas DataFrame (n * p dimension)
    #       x.values: [[x11, x12, ..., x1p], ..., [xn1, ..., xnp]]
    #   y:  
    #       label vector data set
    #       type: Pandas DataFrame (n * 1dimension)
    #       y.values: [[y1], [y2], [y3], ..., [yn]]
    #######################################################################
    def train(self, x, y):
        # deep copy
        train_x = x[:]
        train_y = y[:]

        # apply z-score
        if(self.zscore_):
            train_x = utils.applyZScore(train_x)

        self.model_.fit(train_x.values, train_y.values.ravel())

    #######################################################################
    # Model Prediction Function
    # Input:
    #   x:  
    #       feature vector data set
    #       type: Pandas DataFrame (n * p dimension)
    #       x.values: [[x11, x12, ..., x1p], ..., [xn1, ..., xnp]]
    #
    # Output:
    #   predicted_y:
    #       predicted label vector
    #       type: numpy array (n dimension) [y1, y2, ..., yn]
    #######################################################################
    def predict(self, x):
        # deep copy
        test_x = x[:]

        # apply z-score
        if(self.zscore_):
            test_x = utils.applyZScore(test_x)

        predicted_y = self.model_.predict(test_x.values)

        return predicted_y

    #######################################################################
    # Model Testing Function
    #
    # This function used K-Fold to test the performance of 
    # current learning model, calculate Precision, Recall and Accuracy
    # at each iteration.
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
    #       K-Fold parameter
    #######################################################################
    def tester(self, x, y, k):
        pre_sum, recall_sum, acc_sum = 0, 0, 0

        # generate K-Fold testing data set
        cv = utils.KfoldGenerator(x, y, k)

        for train_x, train_y, test_x, test_y in cv:
            self.train(train_x, train_y)

            predicted_y = self.predict(test_x)

            print(predicted_y)

            precision = utils.computePrecision(predicted_y, test_y.values.ravel())
            recall = utils.computeRecall(predicted_y, test_y.values.ravel())
            accuracy = utils.computeAccuracy(predicted_y, test_y.values.ravel())

            pre_sum += precision
            recall_sum += recall
            acc_sum += accuracy

            print("Precision:", precision)
            print("Recall:", recall)
            print("Accuracy:", accuracy)
            print("-" * 50)

        print("Average Precision:", pre_sum / k)
        print("Average Recall:", recall_sum / k)
        print("Average Accuracy:", acc_sum / k)

def main():
    AAPL = sp.stockParser(utils.AAPL_PATH)

    x, y, date = AAPL.getFluctuationVector(5)
    y = utils.sign(y)

    lm = MLNN(hidden_layers=(5,), zscore=True)
    lm.tester(x, y, 5)

if __name__ == '__main__':
    main()