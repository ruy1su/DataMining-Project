# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import utils
import stockParser as sp

#############################################################
# 
#############################################################
class regressionModel(object):
    #############################################################
    # Constructor
    # Input:
    #   model_type:
    #       0 - linear model; 1 - non-linear model
    #   gradient_type:
    #       0 - closed form; 1 - batch gradient descent
    #   zscore: 
    #       False - do NOT apply z-score normalization
    #       True - apply z-score normalization
    #############################################################
    def __init__(self, model_type, gradient_type=0, zscore=False):
        self.model_type_ = model_type
        self.zscore_ = zscore
        self.gradient_type_ = gradient_type

        # gradient descent learning rate
        self.alpha_ = 0.001

    #############################################################
    # Model Training function
    # Input:
    #   x:  
    #       feature vector data set
    #       type: Pandas DataFrame (n * p dimension)
    #       x.values: [[x11, x12, ..., x1p], ..., [xn1, ..., xnp]]
    #   y:  
    #       label vector data set
    #       type: Pandas DataFrame (n * 1 dimension)
    #       y.values: [[y1], [y2], [y3], ..., [yn]]
    #
    # Output:
    #   self.beta_: 
    #       weight vector
    #       type: np.ndarray (p + 1 dimension)
    #       [b0, b1, b2, ..., bp] 
    #############################################################
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
            # train_x, train_y is DataFrame, convert them into numpy array
            self.beta_ = gradient_func(train_x.values, train_y.values.ravel())

    #############################################################
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
    #       type: numpy array (n dimension)
    #       [y1, y2, y3, ..., yn]
    #############################################################
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

    #############################################################
    # Model Testing Function
    #
    # This function used K-Fold to test the performance of 
    # current learning model, and calculate Most Square Error (MSE)
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
    #############################################################
    def tester(self, x, y, k):
        mse_sum, cnt = 0, 1

        cv = utils.KfoldGenerator(x, y, k)

        for (train_x, train_y, test_x, test_y) in cv:
            print("%s%d-Fold%s" % ("*", cnt, "*"))
            cnt += 1

            self.train(train_x, train_y)

            predicted_y = self.predict(test_x)

            # computeMSE need input params in excatly same dimension
            # Here both predicted_y and test_y.values.ravel() is a
            # n-dimension vector [y1, y2, ..., yn]
            mse = utils.computeMSE(predicted_y, test_y.values.ravel())
            mse_sum += mse

            print("MSE:", mse)

        print("%s%s%s" % ("*", "Average", "*"))
        print("Average MSE:", mse_sum / k)
        

    #############################################################
    # Using closed form formula to calculate the weight vector beta
    #
    # Input:
    #   train_x:  
    #       feature vector data set
    #       type: numpy array (n * p dimension)
    #       train_x: [[x11, x12, ..., x1p], ..., [xn1, ..., xnp]]
    #   train_y:  
    #       label vector data set
    #       type: numpy array (n dimension)
    #       train_y: [y1, y2, y3, ..., yn]
    #
    # Output:
    #   beta:
    #       weight vector (p+1 dimension)
    #       [b0, b1, b2, ..., bn]
    #############################################################
    def closedForm(self, train_x, train_y):
        beta = np.zeros(train_x.shape[1])

        XT = np.transpose(train_x)
        XTX = np.dot(XT, train_x)
        XTY = np.dot(XT, train_y)
        beta = np.dot(np.linalg.inv(XTX), XTY)

        return beta


    #############################################################
    # Using batch gradient descent to calculate the weight vector beta
    #
    # Input:
    #   train_x:  
    #       feature vector data set
    #       type: numpy array (n * p dimension)
    #       train_x: [[x11, x12, ..., x1p], ..., [xn1, ..., xnp]]
    #   train_y:  
    #       label vector data set
    #       type: numpy array (n dimension)
    #       train_y: [y1, y2, y3, ..., yn]
    #
    # Output:
    #   beta:
    #       weight vector (p+1 dimension)
    #       [b0, b1, b2, ..., bn]
    #############################################################
    def getBetaBatchGradient(self, train_x, train_y):
        beta = np.zeros(train_x.shape[1])

        xTrans = train_x.transpose()
        prev_cost = 0

        while(True):
            hypothesis = np.dot(train_x, beta)

            # calculate the difference between predicted value
            # and the truth value
            loss = hypothesis - train_y

            gradient = np.dot(xTrans, loss)

            # calculate the cost function
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

    lm = regressionModel(0, gradient_type=0, zscore=True)
    lm.tester(x, y, 5)


if __name__ == '__main__':
    main()