#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Plotter import Plotter
import tensorflow as tf
import numpy as np
import tester
import featureExtractor as fe


class TensorFlowNN:
    def __init__(self, stepSize=0.1, activation_function=None, hiddenLayers=1, layerNodes=10):
        self.stepSize = stepSize
        self.activation_function = activation_function
        self.hiddenLayers = hiddenLayers
        self.layerNodes = layerNodes

    def train(self, X, Y, iterations=3000):
        # X data (n, d)
        # Y data (n, 1)
        D = X.shape[1]         # dimension of each date point
        X, Y = X[:], Y[:]
        self.x = tf.placeholder(tf.float32, shape=(None, D))
        self.y = tf.placeholder(tf.float32, shape=(None, 1))
        self.Y = Y

        # define layers
        assert(self.hiddenLayers > 0)
        layer = self.add_layer(self.x, D, self.layerNodes)   # input layer
        for i in range(self.hiddenLayers):
            layer = self.add_layer(layer, self.layerNodes, self.layerNodes)   # hidden layers
        self.output = self.add_layer(layer, self.layerNodes, 1)

        # define loss and train process
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.output), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(self.stepSize).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        print('----------------------------')
        for i in range(iterations):
            self.sess.run(self.train_step, feed_dict={self.x: X, self.y: self.Y})
            if (i + 1) % 1000 == 0:
                loss = self.sess.run(self.loss, feed_dict={self.x: X, self.y: self.Y})
                print('iter: ' + str(int((i + 1)/1000)) + 'k | err: ' + str(loss))

        Plotter.plot(self.predict(X), Y)
        print('----------------------------\n')

    def predict(self, X):
        predict = self.sess.run(self.output, feed_dict={self.x: X})
        # print(self.Y.shape)
        return predict.ravel()

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
    # X = np.random.rand(1000, 7)
    # # X = np.linspace(-1, 1, 10)[:, np.newaxis]
    # Y = f(X)
    # Y += np.random.normal(-0.01, 0.01, Y.shape)     # add noise

    if True:
        extractor = fe.featureExtractor(0)
        x, y, date = extractor.getFeature(5, 0)

        nn1 = TensorFlowNN(stepSize=0.1, activation_function=tf.sigmoid, hiddenLayers=3, layerNodes=5)
        # nn2 = TensorFlowNN(stepSize=0.1, activation_function=tf.sigmoid, hiddenLayers=20, layerNodes=50)
        # nn3 = TensorFlowNN(stepSize=0.1, activation_function=tf.sigmoid, hiddenLayers=30, layerNodes=80)
        ts = tester.Tester(2)
        ts.test(nn1, x, y, 0)
        # ts.test(nn2, x, y, 0)
        # ts.test(nn3, x, y, 0)




