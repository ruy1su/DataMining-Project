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
        elif(self.gradient_type_ == 1):
            gradient_func = self.getBetaBatchGradient
        else:
            print("Incorrect gradient type!\n\
                Usage: 0 - closed form solution\n")

        # linear regression
        if(self.model_type_ == 0):
            self.beta_ = gradient_func(train_x.values, train_y.values.ravel())

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

        cv = utils.KfoldGenerator(x, y, k)

        for (train_x, train_y, test_x, test_y) in cv:
            self.train(train_x, train_y)

            predicted_y = self.predict(test_x)
            
            mse = utils.computeMSE(predicted_y, test_y.values.ravel())
            mse_sum += mse

            print("MSE:", mse)

        print("-" * 50)
        print("Average MSE:", mse_sum / k)
        


    def closedForm(self, train_x, train_y):
        beta = np.zeros(train_x.shape[1])

        XT = np.transpose(train_x)
        XTX = np.dot(XT, train_x)
        XTY = np.dot(XT, train_y)
        beta = np.dot(np.linalg.inv(XTX), XTY)

        return beta


    def getBetaBatchGradient(self, train_x, train_y):
        beta = np.zeros(train_x.shape[1])

        xTrans = train_x.transpose()
        prev_cost = 0

        while(True):
            hypothesis = np.dot(train_x, beta)

            loss = hypothesis - train_y

            gradient = np.dot(xTrans, loss)

            cost = np.sum(loss ** 2) / (2 * train_x.shape[0])
            # When cost function does not change, then stop the loop
            if(abs(prev_cost - cost) < 0.00001):
                break
            prev_cost = cost

            # update
            beta = beta - self.alpha_ * gradient

        return beta

def main():
    AAPL = sp.stockParser(utils.AAPL_PATH)

    x, y, date = AAPL.getFluctuationVector(5)

    lm = regressionModel(0, gradient_type=1, zscore=True)
    lm.tester(x, y, 5)


if __name__ == '__main__':
    main()