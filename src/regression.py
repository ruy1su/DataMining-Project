# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn import cross_validation

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


    def tester(self, x, y, k):
        mse_sum = 0

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

            self.train(train_x, train_y)

            predicted_y = self.predict(test_x)

            mse = utils.computeMSE(predicted_y, test_y.values)
            mse_sum += mse

            print("MSE:", mse)

        print("Average MSE:", mse_sum / k)
        


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

    lm = regressionModel(0, zscore=True)
    lm.tester(x, y, 5)


if __name__ == '__main__':
    main()