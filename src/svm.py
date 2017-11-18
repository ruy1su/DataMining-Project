import pandas as pd
import numpy as np

from sklearn import svm

import utils
import stockParser as sp


class SVM(object):
    #######################################################################
    # Constructor
    #
    # Input:
    #   model_type:
    #       0 - linear SVM
    #       1 - Gaussian Radial Basis (RBF) kernel SVM
    #   zscore:
    #       False: do NOT apply z-score normalization
    #       True: apply z-score normalization
    #   class_weight: 
    #       which is a dictionary of form {class_label : value},
    #       where `value` is a floating point number > 0 that sets the 
    #       parameter C of class class_label to C * value. The greater
    #       the `value`, the more chance that a sample point of same 
    #       class will be classified into its own class (recall up, but
    #       precision may go down). 
    #######################################################################
    def __init__(self, model_type, zscore=False, class_weight={}):
        self.model_type_ = model_type
        self.zscore_ = zscore
        self.class_weight_ = class_weight

        if(model_type == 0):
            self.model_ = svm.SVC(kernel='linear', class_weight=class_weight)
        elif(model_type == 1):
            self.model_ = svm.SVC(kernel='rbf', class_weight=class_weight)

    #######################################################################
    # Model Training function
    # Input:
    #   x:  
    #       feature vector data set
    #       type: Pandas DataFrame (n * p dimension)
    #       x.values: [[x11, x12, ..., x1p], ..., [xn1, ..., xnp]]
    #   y:  
    #       label vector data set
    #       type: Pandas DataFrame (n * 1 dimension)
    #       y.values: [[y1], [y2], [y3], ..., [yn]]
    #######################################################################
    def train(self, x, y):
        # deep copy
        train_x = x[:]
        train_y = y[:]

        # apply z-score
        if(self.zscore_):
            train_x = utils.applyZScore(train_x)

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
    #       type: numpy array (n dimension)
    #       [y1, y2, y3, ..., yn]
    #######################################################################
    def predict(self, x):
        # deep copy
        test_x = x[:]

        # apply z-score
        if(self.zscore_):
            test_x = utils.applyZScore(test_x)

        predicted_y = self.model_.predict(test_x.values)

        return predicted_y

    #######################################################################
    # Model Testing Function
    #
    # This function used K-Fold to test the performance of 
    # current learning model, calculate Precision, Recall and Accuracy
    # at each iteration.
    #
    # Input:
    #   x:  
    #       feature vector data set
    #       type: Pandas DataFrame (n * p dimension)
    #       x.values: [[x11, x12, ..., x1p], ..., [xn1, ..., xnp]]
    #   y:  
    #       label vector data set
    #       type: Pandas DataFrame (n * 1 dimension)
    #       y.values: [[y1], [y2], [y3], ..., [yn]]
    #   k:
    #       K-Fold parameter
    #######################################################################
    def tester(self, x, y, k):
        pre_sum, recall_sum, acc_sum, cnt = 0, 0, 0, 1

        # generate K-Fold testing data set
        cv = utils.KfoldGenerator(x, y, k)

        for train_x, train_y, test_x, test_y in cv:
            print("%s%d-Fold%s" % ("*", cnt, "*"))
            cnt += 1

            self.train(train_x, train_y)

            predicted_y = self.predict(test_x)

            precision = utils.computePrecision(predicted_y, test_y.values.ravel())
            recall = utils.computeRecall(predicted_y, test_y.values.ravel())
            accuracy = utils.computeAccuracy(predicted_y, test_y.values.ravel())

            pre_sum += precision
            recall_sum += recall
            acc_sum += accuracy

            print("Precision:", precision)
            print("Recall:", recall)
            print("Accuracy:", accuracy)

        print("%s%s%s" % ("*", "Average", "*"))
        print("Average Precision:", pre_sum / k)
        print("Average Recall:", recall_sum / k)
        print("Average Accuracy:", acc_sum / k)

def main():
    AAPL = sp.stockParser(utils.AAPL_PATH)

    x, y, date = AAPL.getFluctuationVector(5)
    y = utils.sign(y)

    lm = SVM(1, zscore=True)
    lm.tester(x, y, 5)

if __name__ == '__main__':
    main()