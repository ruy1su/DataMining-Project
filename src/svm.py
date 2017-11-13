import pandas as pd
import numpy as np

from sklearn import svm

import utils
import stockParser as sp


class SVM(object):
    #############################################################
    # 
    # class_weight: is a dictionary of form {class_label : value},
    #   where `value` is a floating point number > 0 that sets the 
    #   parameter C of class class_label to C * value. The greater
    #   the `value`, the more chance that a sample point of same 
    #   class will be classified into its own class (recall up, but
    #   precision may go down). 
    #############################################################
    def __init__(self, model_type, zscore=False, class_weight={}):
        self.model_type_ = model_type
        self.zscore_ = zscore
        self.class_weight_ = class_weight

        if(model_type == 0):
            self.model_ = svm.SVC(kernel='linear', class_weight=class_weight)
        elif(model_type == 1):
            self.model_ = svm.SVC(kernel='rbf', class_weight=class_weight)


    def train(self, x, y):
        # deep copy
        train_x = x[:]
        train_y = y[:]

        # apply z-score
        if(self.zscore_):
            train_x = utils.applyZScore(train_x)

        self.model_.fit(train_x.values, train_y.values.ravel())

    def predict(self, x):
        # deep copy
        test_x = x[:]

        # apply z-score
        if(self.zscore_):
            test_x = utils.applyZScore(test_x)

        predicted_y = self.model_.predict(test_x)

        # [0, 1, 0, 1] --> [[0],[1],[0],[1]]
        predicted_y = np.array([[predicted_y[i]] for i in range(len(predicted_y))])

        return predicted_y

    def tester(self, x, y, k):
        pre_sum, recall_sum, acc_sum = 0, 0, 0

        cv = utils.KfoldGenerator(x, y, k)

        for train_x, train_y, test_x, test_y in cv:
            self.train(train_x, train_y)

            predicted_y = self.predict(test_x)

            precision = utils.computePrecision(predicted_y.ravel(), test_y.values.ravel())
            recall = utils.computeRecall(predicted_y.ravel(), test_y.values.ravel())
            accuracy = utils.computeAccuracy(predicted_y.ravel(), test_y.values.ravel())

            pre_sum += precision
            recall_sum += recall
            acc_sum += accuracy

            print("Precision:", precision)
            print("Recall:", recall)
            print("Accuracy:", accuracy)
            print("-" * 50)

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