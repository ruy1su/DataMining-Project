# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import utils
import stockParser as sp

class regressionModel(object):
    def __init__(self, model_type, gradient_type=0, zscore=False):
        self.model_type_ = model_type
        self.zscore_ = zscore
        self.gradient_type_ = gradient_type
        self.alpha_ = 0.001


    def train(self, x, y):
        # deep copy
        train_x = x[:]
        train_y = y[:]

        # apply z-score
        if(self.zscore_):
            train_x = utils.applyZScore(train_x)

        # Add bias column at feature vector
        train_x.insert(0, 'offset', 1)

        # choose gradient descent function
        gradient_func = None
        if(self.gradient_type_ == 0):
            gradient_func = self.closedForm
        else:
            print("Incorrect gradient type!\n\
                Usage: 0 - closed form solution\n")

        # linear regression
        if(self.model_type_ == 0):
            self.beta_ = gradient_func(train_x, train_y)


    def predict(self, x):
        # deep copy
        test_x = x[:]

        # apply z-score
        if(self.zscore_):
            test_x = utils.applyZScore(test_x)

        # Add bias column at feature vector
        test_x.insert(0, 'offset', 1)

        # linear regression
        if(self.model_type_ == 0):
            predicted_y = test_x.values.dot(self.beta_)

        return predicted_y


    def test(self, test_x, test_y):
        predicted_y = self.predict(test_x)
        mse = utils.computeMSE(predicted_y, test_y.values)
        print("MSE:", mse)


    def closedForm(self, train_x, train_y):
        beta = np.zeros(train_x.shape[1])

        XT = np.transpose(train_x)
        XTX = np.dot(XT, train_x)
        XTY = np.dot(XT, train_y)
        beta = np.dot(np.linalg.inv(XTX), XTY)

        return beta

def main():
    AAPL = sp.stockParser(utils.AAPL_PATH)

    x, y, date = AAPL.getFluctuationVector(5)

    utils.KfoldTester(regressionModel(0, zscore=True), x, y, 5)

    # lm = regressionModel(0, zscore=True)
    # lm.train(x, y)
    # lm.test(x, y)

    # lm = regressionModel(0, zscore=False)
    # lm.train(x, y)
    # lm.test(x, y)

if __name__ == '__main__':
    main()