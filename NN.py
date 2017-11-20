#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class NN:
    def __init__(self, X, Y, stepSize):
        self.X = X
        self.Y = Y

        # define placeholder for inputs to network  
        self.xs = tf.placeholder(tf.float32, [None, 1])
        self.ys = tf.placeholder(tf.float32, [None, 1])

        # 3.定义神经层：隐藏层和预测层
        l1 = self.add_layer(self.xs, 1, 10, activation_function=tf.nn.relu)
        l2 = self.add_layer(l1, 10, 10, activation_function=tf.nn.relu)
        # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
        output = self.add_layer(l2, 10, 1, activation_function=None)

        # 4.定义 loss 表达式
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - output), reduction_indices=[1]))

        # 5.选择 optimizer 使 loss 达到最小                   
        # 这一行定义了用什么方式去减少 loss，学习率是 0.1       
        self.train_step = tf.train.GradientDescentOptimizer(stepSize).minimize(self.loss)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        # 上面定义的都没有运算，直到 sess.run 才会开始运算
        self.sess.run(init)

    # add one more layer and return the output of this layer
    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def train(self):
        for i in range(1000):
            # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
            self.sess.run(self.train_step, feed_dict={self.xs: self.X, self.ys: self.Y})
            if i % 50 == 0:
                self.predict()

    def predict(self):
        print(self.sess.run(self.loss, feed_dict={self.xs: self.X, self.ys: self.Y}))


if __name__ == '__main__':
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise
    nn = NN(x_data, y_data, stepSize=0.1)
    nn.train()




