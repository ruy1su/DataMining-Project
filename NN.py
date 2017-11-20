#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class NN:
    def __init__(self, X, Y, stepSize=0.1, activation_function=None, hiddenLayers=1, layerNodes=10):
        self.X = X
        self.Y = Y
        self.stepSize = stepSize
        self.activation_function = activation_function
        # placeholders for inputs
        self.x = tf.placeholder(tf.float32, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])

        # define layers
        assert(hiddenLayers > 0)
        layer = self.add_layer(self.x, 1, layerNodes)   # input layer
        for i in range(hiddenLayers):
            layer = self.add_layer(layer, layerNodes, layerNodes)   # add hidden layers
        output = self.add_layer(layer, layerNodes, 1)

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - output), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(stepSize).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def train(self, iterations):
        for i in range(iterations):
            # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
            self.sess.run(self.train_step, feed_dict={self.x: self.X, self.y: self.Y})
            if i % 100 == 0:
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


if __name__ == '__main__':
    X = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, X.shape)
    Y = np.square(X) - 0.5 + noise

    nn = NN(X, Y, stepSize=0.1, activation_function=tf.sigmoid, hiddenLayers=5, layerNodes=30)
    nn.train(10000)




