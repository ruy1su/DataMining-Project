# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from Plotter import Plotter
import stockParser as sp
import featureExtractor as fe
import regression as reg
import svm
import mlnn
import utils

# define companies here
TEST_COMPANY = ["Google", "Apple", "Microsoft"]

class Tester(object):
    #######################################################################
    # Constructor
    #
    # Input:
    #   k_fold: Integer
    #       specify the param for cross validation
    #       how many folds you want to use here
    #######################################################################
    def __init__(self, k_fold=5):
        self.k_fold_ = k_fold


    #############################################################
    # Model Testing Function
    #
    # This function used K-Fold to test the performance of 
    # current learning model, and calculate Most Square Error (MSE)
    # at each iteration.
    #
    # Input:
    #   model:
    #       learning model, must have interface `train(x, y)`
    #       and interface `test(x)`
    #   x:  
    #       feature vector data set
    #       type: Pandas DataFrame (n * p dimension)
    #       x.values: [[x11, x12, ..., x1p], ..., [xn1, ..., xnp]]
    #   y:  
    #       label vector data set
    #       type: Pandas DataFrame (n * 1 dimension)
    #       y.values: [[y1], [y2], [y3], ..., [yn]]
    #   mode:
    #       Tester Mode
    #       0 - regression testing
    #       1 - classifier testing
    #############################################################
    def test(self, model, x, y, mode):
        predicted_y_set = []
        test_y_set = []

        # generate k-fold testing and training set
        cv = utils.KfoldGenerator(x, y, self.k_fold_)

        for (train_x, train_y, test_x, test_y) in cv:
            model.train(train_x, train_y)

            # predicted_y: [y1, y2, y3, ..., yn]
            predicted_y = model.predict(test_x)
            predicted_y_set.append(predicted_y)

            # test_y: DataFrame [[y1], [y2], [y3], ..., [yn]]
            # test_y.values.ravel(): [y1, y2, y3, ..., yn]
            test_y_set.append(test_y.values.ravel())

        # measure the prediction results
        self.measure(predicted_y_set, test_y_set, mode)


    #######################################################################
    # Measure the prediction results
    # 
    # This function measure the prediction for regression and 
    # classification tasks. For regression, it measures the Mean Square
    # Error (MSE). For classification, it measures precision, recall and
    # accuracy.
    #
    # Input:
    #   predicted_ys:
    #       A list of predict labels, each sublist is a list of predict
    #       labels for one single test case.
    #       [[y1, y2, ..., yn], [y1, y2, ..., yn], ...]
    #   test_ys:
    #       A list of ground-true labels, each sublist is a list of ground-
    #       true labels for one single test case.
    #       [[y1, y2, ..., yn], [y1, y2, ..., yn], ...]
    #   mode:
    #       Tester Mode
    #       0 - regression testing
    #       1 - classifier testing
    #######################################################################
    def measure(self, predicted_ys, test_ys, mode):
        mse_sum, pre_sum, recall_sum, acc_sum, cnt = 0, 0, 0, 0, 1

        for predicted_y, test_y in zip(predicted_ys, test_ys):
            print("%s%d-Fold%s" % ("*", cnt, "*"))
            cnt += 1

            # regression measurement
            if(mode == 0):
                # computeMSE need input params in excatly same dimension
                # Here both predicted_y and test_y.values.ravel() is a
                # n-dimension vector [y1, y2, ..., yn]
                mse = utils.computeMSE(predicted_y, test_y)
                mse_sum += mse

                # print("Predict_y", predicted_y);
                # print("Test_y", test_y);

                print("MSE:", mse)

            # classification measurement
            elif(mode == 1):
                precision = utils.computePrecision(predicted_y, test_y)
                recall = utils.computeRecall(predicted_y, test_y)
                accuracy = utils.computeAccuracy(predicted_y, test_y)

                pre_sum += precision
                recall_sum += recall
                acc_sum += accuracy

                print("Precision:", precision)
                print("Recall:", recall)
                print("Accuracy:", accuracy)

            print (test_y[:20], predicted_y[:20])
            Plotter.plot(predicted_y, test_y)

        # print out average
        print("%s%s%s" % ("*", "Average", "*"))
        if(mode == 0):
            print("Average MSE:", mse_sum / self.k_fold_)
        elif(mode == 1):
            print("Average Precision:", pre_sum / self.k_fold_)
            print("Average Recall:", recall_sum / self.k_fold_)
            print("Average Accuracy:", acc_sum / self.k_fold_)


    #######################################################################
    # Test ALL models
    #
    # This function will test all models with every dataset (AAPL, GOOG, MSFT)
    #
    # Input:
    #   fluc: integer, length of stock fluctuation vector
    #   sentiment: integer, specify sentiment vector (TBD) 
    #######################################################################
    def testAll(self, fluc=5, sentiment=0):
        # linear regression model
        model_lreg = reg.regressionModel(0, gradient_type=0, zscore=True)
        model_svm = svm.SVM(1, zscore=True)
        model_nn = mlnn.MLNN(hidden_layers=(5,), zscore=True)
        
        for i in range(0, len(TEST_COMPANY)):
            # create feature extractor for current company
            extractor = fe.featureExtractor(i)

            # get features, label and corresponding date
            x, y, date = extractor.getFeature(fluc, sentiment)
            # convert numerical label into discrete value for classifier
            discrete_y = utils.sign(y)

            # Test Linear Regression
            print("# Linear Regression Tester with " + TEST_COMPANY[i])
            self.test(model_lreg, x, y, 0)
            print("-" * 60)

            # Test SVM
            print("# SVM Tester with " + TEST_COMPANY[i])
            self.test(model_svm, x, discrete_y, 1)
            print("-" * 60)

            # Test multi-layer nueral network
            print("# Neural Network Tester with " + TEST_COMPANY[i])
            self.test(model_nn, x, discrete_y, 1)
            print("-" * 60)

def main():
    tester = Tester(5)
    tester.testAll(5, 0)

if __name__ == '__main__':
    main()