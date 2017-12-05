#!/usr/local/bin/python3

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf

tf = False
TEST_ALL_MODEL = False
FEATURE_COMPARISON = True

from utils import tester

def main():
    # initial tester k-fold
    ts = tester.Tester(2)

    # Tensorflow Tester
    if(tf == True):
        ts.testTensorFlowRandom(stepSize=0.01, hiddenLayers=2, layerNodes=2, activation_function=tf.tanh)
        ts.testTensorFlow(stepSize=0.01, hiddenLayers=2, layerNodes=2, activation_function=tf.tanh)

    # Test all models with same feature selection and dataset
    if(TEST_ALL_MODEL):
        ts.testAllModels(1, 1)

    if(FEATURE_COMPARISON):
        print("-" * 60)
        print("1-day fluctuation, sentimental analysis positive/negative")
        ts.testSingleModel(1, 0, 180)
        print("-" * 60)

        print("3-day fluctuation, sentimental analysis positive/negative")
        ts.testSingleModel(3, 0, 180)
        print("-" * 60)

        print("5-day fluctuation, sentimental analysis positive/negative")
        ts.testSingleModel(5, 0, 180)
        print("-" * 60)

        print("1-day fluctuation, sentimental analysis multiple moods")
        ts.testSingleModel(1, 1, 180)
        print("-" * 60)

        print("3-day fluctuation, sentimental analysis multiple moods")
        ts.testSingleModel(3, 1, 180)
        print("-" * 60)

        print("5-day fluctuation, sentimental analysis multiple moods")
        ts.testSingleModel(5, 1, 180)
        print("-" * 60)

if __name__ == '__main__':
    main()