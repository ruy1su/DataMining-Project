#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class NN:
    def __init__(self, X, Y, stepSize=0.1, activation_function=None, hiddenLayers=1, layerNodes=10, iterations=3000):
        self.X = X    # X data (n, d)
        self.Y = Y    # Y data (n, 1)
        self.N = X.shape[0]         # number of data points
        self.D = X.shape[1]         # dimension of each date point
        self.stepSize = stepSize
        self.activation_function = activation_function
        self.x = tf.placeholder(tf.float32, shape=(self.N, self.D))
        self.y = tf.placeholder(tf.float32, shape=(self.N, 1))

        # define layers
        assert(hiddenLayers > 0)
        layer = self.add_layer(self.x, self.D, layerNodes)   # input layer
        for i in range(hiddenLayers):
            layer = self.add_layer(layer, layerNodes, layerNodes)   # hidden layers
        output = self.add_layer(layer, layerNodes, 1)

        # define loss and train process
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - output), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(stepSize).minimize(self.loss)

        # start and train
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.train(iterations)

    def train(self, iterations):
        for i in range(iterations):
            self.sess.run(self.train_step, feed_dict={self.x: self.X, self.y: self.Y})
            if i % 1000 == 0:
                self.predict()

    def predict(self):
        result = self.sess.run(self.loss, feed_dict={self.x: self.X, self.y: self.Y})
        print('err: ' + str(result))

    # add one more layer and return the output of this layer
    def add_layer(self, inputs, in_size, out_size):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if self.activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = self.activation_function(Wx_plus_b)
        return outputs


def f(X):
    beta = np.random.rand(X.shape[1], 1)
    return X.dot(beta)


if __name__ == '__main__':
    X = np.random.rand(1000, 7)
    # X = np.linspace(-1, 1, 10)[:, np.newaxis]
    Y = f(X)
    Y += np.random.normal(-0.01, 0.01, Y.shape)     # add noise

    print X.shape, Y.shape

    nn = NN(X, Y, stepSize=0.1, activation_function=tf.sigmoid, hiddenLayers=2, layerNodes=10)
    print('---------')

    nn = NN(X, Y, stepSize=0.1, activation_function=tf.sigmoid, hiddenLayers=5, layerNodes=30)
    print('---------')

    nn = NN(X, Y, stepSize=0.1, activation_function=tf.sigmoid, hiddenLayers=10, layerNodes=80)



