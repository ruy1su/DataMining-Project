#!/usr/local/bin/python3

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf

TF = False
TEST_ALL_MODEL = True
FEATURE_COMPARISON = True
DATA_SET_SIZE_TEST = False

from utils import tester

def main():
    # initial tester k-fold
    ts = tester.Tester(2)

    # Tensorflow Tester
    if(TF):
        ts.testTensorFlowRandom(stepSize=0.01, hiddenLayers=2, layerNodes=2, activation_function=tf.tanh, noise=0)
        ts.testTensorFlowRandom(stepSize=0.01, hiddenLayers=2, layerNodes=2, activation_function=tf.tanh, noise=0.1)
        ts.testTensorFlowRandom(stepSize=0.01, hiddenLayers=2, layerNodes=2, activation_function=tf.tanh, noise=0.3)
        ts.testTensorFlowRandom(stepSize=0.01, hiddenLayers=2, layerNodes=2, activation_function=tf.tanh, noise=0.5)
        ts.testTensorFlowRandom(stepSize=0.01, hiddenLayers=2, layerNodes=2, activation_function=tf.tanh, noise=0.8)
        # ts.testTensorFlow(stepSize=0.01, hiddenLayers=2, layerNodes=2, activation_function=tf.tanh)

    # Test all models with same feature selection and dataset
    if(TEST_ALL_MODEL):
        ts.testAllModels(1, 1)

    if(FEATURE_COMPARISON):
        data_set_size = 180

        ts.testSingleModel(0, 0, data_set_size)
        ts.testSingleModel(1, 0, data_set_size)
        ts.testSingleModel(3, 0, data_set_size)
        ts.testSingleModel(5, 0, data_set_size)
        ts.testSingleModel(0, 1, data_set_size)
        ts.testSingleModel(1, 1, data_set_size)
        ts.testSingleModel(3, 1, data_set_size)
        ts.testSingleModel(5, 1, data_set_size)
        ts.testSingleModel(0, 2, data_set_size)
        ts.testSingleModel(1, 2, data_set_size)
        ts.testSingleModel(3, 2, data_set_size)
        ts.testSingleModel(5, 2, data_set_size)
        ts.testSingleModel(0, 3, data_set_size)
        ts.testSingleModel(1, 3, data_set_size)
        ts.testSingleModel(3, 3, data_set_size)
        ts.testSingleModel(5, 3, data_set_size)


    if(DATA_SET_SIZE_TEST):
        print("-" * 60)
        print("Dataset Size: 60")
        ts.testSingleModel(1, 1, 60)
        print("-" * 60)

        print("Dataset Size: 120")
        ts.testSingleModel(1, 1, 120)
        print("-" * 60)

        print("Dataset Size: 180")
        ts.testSingleModel(1, 1, 180)
        print("-" * 60)

        print("Dataset Size: 360")
        ts.testSingleModel(1, 1, 360)
        print("-" * 60)

        print("Dataset Size: 720")
        ts.testSingleModel(1, 1, 720)
        print("-" * 60)

        print("Dataset Size: full size")
        ts.testSingleModel(1, 1)
        print("-" * 60)



if __name__ == '__main__':
    main()