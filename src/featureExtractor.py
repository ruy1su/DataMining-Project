import pandas as pd
import numpy as np

import utils
import stockParser as sp

STOCK_PATH = [utils.GOOG_PATH, utils.AAPL_PATH, utils.MSFT_PATH]

class featureExtractor(object):

    def __init__(self, target):
        self.stock_path_ = STOCK_PATH[target]
        self.stock_parser_ = sp.stockParser(self.stock_path_)

    def getFeature(self, fluc = 5, sentiment = 0):
        fluc_vec, y, date = self.stock_parser_.getFluctuationVector(fluc)
        # Get sentiment vector

        # append sentiment vector to stock fluctuation vector
        x = fluc_vec

        return x, y, date




def main():
    model = featureExtractor(0)
    print(model.getFeature())

if __name__ == '__main__':
    main()