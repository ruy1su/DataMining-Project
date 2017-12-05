#!/usr/local/bin/python3

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf

tf = False
TEST_ALL_MODEL = True

from utils import tester

def main():
    # initial tester k-fold
    ts = tester.Tester(2)

    if(tf == True):
        ts.testTensorFlowRandom(stepSize=0.01, hiddenLayers=2, layerNodes=2, activation_function=tf.tanh)
        ts.testTensorFlow(stepSize=0.01, hiddenLayers=2, layerNodes=2, activation_function=tf.tanh)

    if(TEST_ALL_MODEL):
        ts.testAllModels(1, 1)

if __name__ == '__main__':
    main()