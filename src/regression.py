# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import utils
import tester
import stockParser as sp
import featureExtractor as fe

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
    # create feature extractor for Google
    extractor = fe.featureExtractor(0)

    # get features, label and corresponding date
    x, y, date = extractor.getFeature(5, 0)

    ts = tester.Tester(5)
    ts.test(regressionModel(0, gradient_type=0, zscore=True), x, y, 0)


if __name__ == '__main__':
    main()