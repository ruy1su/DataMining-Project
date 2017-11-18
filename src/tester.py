# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import stockParser as sp
import featureExtractor as fe
import regression as reg
import svm
import mlnn
import utils


TEST_CASES = ["Google", "Apple", "Microsoft"]

def tester(k_fold = 5):
    # linear regression model
    model_lreg = reg.regressionModel(0, gradient_type=0, zscore=True)
    model_svm = svm.SVM(1, zscore=True)
    model_nn = mlnn.MLNN(hidden_layers=(5,), zscore=True)
    
    for i in range(0, len(TEST_CASES)):
        # create feature extractor for current company
        extractor = fe.featureExtractor(i)

        x, y, date = extractor.getFeature(5, 0)
        discrete_y = utils.sign(y)

        print("# Linear Regression Tester with " + TEST_CASES[i])
        model_lreg.tester(x, y, k_fold)
        print("-" * 60)

        print("# SVM Tester with " + TEST_CASES[i])
        model_svm.tester(x, discrete_y, k_fold)
        print("-" * 60)

        print("# Neural Network Tester with " + TEST_CASES[i])
        model_nn.tester(x, discrete_y, k_fold)
        print("-" * 60)

def main():
    tester(5)

if __name__ == '__main__':
    main()