# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Reads the data from CSV files, converts it into Dataframe and returns x and y dataframes
def getDataframe(filePath):
    dataframe = pd.read_csv(filePath)
    y = dataframe['y']
    x = dataframe.drop('y', axis=1)
    return x, y

# Applies z-score normalization to the dataframe and returns a normalized dataframe
def applyZScore(dataframe):
    normalized_dataframe = dataframe
    ########## Please Fill Missing Lines Here ##########
    expectation = np.mean(dataframe, axis=0)
    std_dev = np.std(dataframe, axis=0)
    normalized_dataframe = (dataframe - expectation) / std_dev

    return normalized_dataframe

# train_x and train_y are numpy arrays
# function returns value of beta calculated using (0) the formula beta = (X^T*X)^ -1)*(X^T*Y)
def getBeta(train_x, train_y):
    beta = np.zeros(train_x.shape[1])
    ########## Please Fill Missing Lines Here ##########
    XT = np.transpose(train_x)
    XTX = np.dot(XT, train_x)
    XTY = np.dot(XT, train_y)
    beta = np.dot(np.linalg.inv(XTX), XTY)

    return beta
    
# train_x and train_y are numpy arrays
# alpha (learning rate) is a scalar
# function returns value of beta calculated using (1) batch gradient descent
def getBetaBatchGradient(train_x, train_y, alpha):
    beta = np.zeros(train_x.shape[1])
    ########## Please Fill Missing Lines Here ##########

    xTrans = train_x.transpose()
    prev_cost = 0

    while(True):
        hypothesis = np.dot(train_x, beta)
        loss = hypothesis - train_y
        gradient = np.dot(xTrans, loss)

        cost = np.sum(loss ** 2) / (2 * train_x.shape[0])
        # When cost function does not change, then stop the loop
        if(abs(prev_cost - cost) < 0.0000001):
            break
        prev_cost = cost

        # update
        beta = beta - alpha * gradient

    return beta
    
# train_x and train_y are numpy arrays
# alpha (learning rate) is a scalar
# function returns value of beta calculated using (2) stochastic gradient descent
def getBetaStochasticGradient(train_x, train_y, alpha):
    beta = np.zeros(train_x.shape[1])
    ########## Please Fill Missing Lines Here ##########
    prev_cost = 0
    while(True):
        cost = 0
        for i in xrange(0, train_x.shape[0]):
            hypothesis = np.dot(train_x[i].transpose(), beta)
            loss = train_y[i] - hypothesis
            gradient = alpha * np.dot(loss, train_x[i])
            cost += loss ** 2 / (2 * train_x.shape[0])
            # update
            beta = beta + gradient

        # When cost function does not change, then stop the loop
        if(abs(prev_cost - cost) < 0.0001):
            break;
        prev_cost = cost

    return beta

# predicted_y and test_y are the predicted and actual y values respectively as numpy arrays
# function prints the mean squared error value for the test dataset
def compute_mse(predicted_y, test_y):
    mse = 100
    ########## Please Fill Missing Lines Here ##########
    mse = np.sum((predicted_y - test_y) ** 2) / predicted_y.shape[0]

    print 'MSE: ', mse
    
# Linear Regression implementation
class LinearRegression(object):
    # Initializes by reading data, setting hyper-parameters, and forming linear model
    # Forms a linear model (learns the parameter) according to type of beta (0 - closed form, 1 - batch gradient, 2 - stochastic gradient)
    # Performs z-score normalization if z_score is 1
    def __init__(self, beta_type, z_score = 0):
        self.alpha = 0.01
        self.beta_type = beta_type
        self.z_score = z_score

        self.train_x, self.train_y = getDataframe('linear-regression-train.csv')
        self.test_x, self.test_y = getDataframe('linear-regression-test.csv')
        
        if(z_score == 1):
            self.alpha = 0.001
            self.train_x = applyZScore(self.train_x)
            self.test_x = applyZScore(self.test_x)
        else:
            self.alpha = 0.00001
        
        # Prepend columns of 1 for beta 0
        self.train_x.insert(0, 'offset', 1)
        self.test_x.insert(0, 'offset', 1)
        
        self.linearModel()
    
    # Gets the beta according to input
    def linearModel(self):
        if(self.beta_type == 0):
            self.beta = getBeta(self.train_x.values, self.train_y.values)
            print 'Beta: '
            print self.beta
        elif(self.beta_type == 1):
            self.beta = getBetaBatchGradient(self.train_x.values, self.train_y.values, self.alpha)
            print 'Beta: '
            print self.beta
        elif(self.beta_type == 2):
            self.beta = getBetaStochasticGradient(self.train_x.values, self.train_y.values, self.alpha)
            print 'Beta: '
            print self.beta
        else:
            print 'Incorrect beta_type! Usage: 0 - closed form solution, 1 - batch gradient descent, 2 - stochastic gradient descent'
            
    # Predicts the y values of all test points
    # Outputs the predicted y values to the text file named "linear-regression-output_betaType_zScore" inside "output" folder
    # Computes MSE
    def predict(self):
        self.predicted_y = self.test_x.values.dot(self.beta)
        np.savetxt('output/linear-regression-output' + '_' + str(self.beta_type) + '_' + str(self.z_score) + '.txt', self.predicted_y)
        compute_mse(self.predicted_y, self.test_y.values)
        
if __name__ == '__main__':
    # Change 1st paramter to 0 for closed form, 1 for batch gradient, 2 for stochastic gradient
    # Add a second paramter with value 1 for z score normalization
    print '------------------------------------------------'
    print 'Closed Form Without Normalization'
    lm = LinearRegression(0)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Batch Gradient Without Normalization'
    lm = LinearRegression(1)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Stochastic Gradient Without Normalization'
    lm = LinearRegression(2)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Closed Form With Normalization'
    lm = LinearRegression(0, 1)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Batch Gradient With Normalization'
    lm = LinearRegression(1, 1)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Stochastic Gradient With Normalization'
    lm = LinearRegression(2, 1)
    lm.predict()
    print '------------------------------------------------'