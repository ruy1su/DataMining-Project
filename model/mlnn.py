# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from utils import tools
from utils import tester
from utils import stockParser as sp
from utils import featureExtractor as fe


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
            train_x = tools.applyZScore(train_x)

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
            test_x = tools.applyZScore(test_x)

        predicted_y = self.model_.predict(test_x.values)

        return predicted_y


def main():
    # create feature extractor for Google
    extractor = fe.featureExtractor(0)

    # get features, label and corresponding date
    x, y, date = extractor.getFeature(1, 1)
    # y = tools.sign(y)

    ts = tester.Tester(5)
    ts.test(MLNN(hidden_layers=(2, 2), zscore=False), x, y, 0)

if __name__ == '__main__':
    main()