#!/usr/local/bin/python3

import pandas as pd
import numpy as np
import utils

# If DEBUG is True, will print out DEBUG info
DEBUG = False

class stockParser(object):

    def __init__(self, file_path):
        self.data_frame_ = pd.read_csv(file_path)
        self.size_ = len(self.data_frame_.values)

        if DEBUG:
            self.data_frame_ = self.data_frame_.loc[:10,:]
            self.size_ = len(self.data_frame_.values)

            print (self.data_frame_)

    #############################################################
    # Get fluctuation vectors which contain fluctuation rate of
    # past k days.
    #
    # Input: k - demensions of vector
    # Output: x, y, date - Pandas DataFrame
    #         x is fluctuation rate (k-demensional)
    #         y is label (fluctuation rate of next day)
    #         date is current date of vector x
    #############################################################
    def getFluctuationVector(self, k):
        if k < 0:
            return None

        # size of fluc_record = size_ - 1, it ripped out the head of data_frame_
        fluc_record = [self.calcFluctuation(i - 1, i) for i in range(1, self.size_)]

        fluc_vec = []
        for i in range(0, k):
            fluc_vec.append(fluc_record[k - i - 1 : len(fluc_record) - i - 1])
            # fluc_vec.append([fluc_record[j - i] for j in range(k - 1, len(fluc_record) - 1)])

        # convert to dict for DataFrame initialization
        fluc_vec = {'x'+str(i):fluc_vec[i - 1] for i in range(1, k + 1)}

        # Add label for each vector
        # fluc_vec['y'] = fluc_record[k : ]
        fluc_vec['y'] = [fluc_record[i] * 100 for i in range(k, len(fluc_record))]

        # Add Date associate with each vector
        fluc_vec['Date'] = self.data_frame_.loc[k : self.size_ - 2, 'Date'].values
        # fluc_vec['Date'] = [self.data_frame_.loc[i, 'Date'] for i in range(k, self.size_ - 1)]
        
        # convert to Pandas DataFrame
        fluc_vec = pd.DataFrame(fluc_vec)


        # separate DataFrame into x, y, date
        x = fluc_vec.drop(['y', 'Date'], axis=1)
        y = fluc_vec['y']
        date = fluc_vec['Date']

        if DEBUG:
            print (fluc_record)
            print (fluc_vec)
            print (x)
            print (y)
            print (date)

        return x, y, date

    #############################################################
    # Calculate the fluctuation based on closing price of current
    # day and closing price of previous day.
    # 
    # Input: index of prevous and current stock
    #############################################################
    def calcFluctuation(self, prev_ix, cur_ix):
        prev = self.data_frame_.loc[prev_ix, 'Close']
        cur = self.data_frame_.loc[cur_ix, 'Close']

        return (cur - prev) / prev


    def test(self):
        pass


def main():
    stock_parser = stockParser(GOOG_PATH)
    stock_parser.test()


if __name__ == '__main__':
    main()