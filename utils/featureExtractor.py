import pandas as pd
import numpy as np

from utils import tools
from utils import stockParser as sp

# define path to stock records
STOCK_PATH = [tools.GOOG_PATH, tools.AAPL_PATH, tools.MSFT_PATH]

# define path to sentimental analysis result (Negative / Positive)
MOOD_PN_PATH = [tools.GOOG_MOOD_PN, tools.AAPL_MOOD_PN, tools.MSFT_MOOD_PN]

# define path to sentimental analysis result (multi-dimension)
MOOD_MULTI_PATH = [tools.GOOG_MOOD_MULTI, tools.AAPL_MOOD_MULTI, tools.MSFT_MOOD_MULTI]

MOOD_PN_CW_PATH = [tools.GOOG_MOOD_PN_CW, tools.AAPL_MOOD_PN_CW, tools.MSFT_MOOD_PN_CW]

MOOD_PN_CW2_PATH = [tools.GOOG_MOOD_PN_CW2, tools.AAPL_MOOD_PN_CW2, tools.MSFT_MOOD_PN_CW2]

# define sentimental analysis path
MOOD_PATH = [MOOD_PN_PATH, MOOD_MULTI_PATH, MOOD_PN_CW_PATH, MOOD_PN_CW2_PATH]


class featureExtractor(object):

    def __init__(self, target):
        self.target_ = target


    def changeTargetCompany(self, target):
        self.target_ = target


    def getFeature(self, fluc = 5, sentiment = 0):
        stock_path = STOCK_PATH[self.target_]
        mood_path = MOOD_PATH[sentiment][self.target_]

        # get stock record
        stock_parser = sp.stockParser(stock_path)
        fluc_vec, fluc_y, fluc_date = stock_parser.getFluctuationVector(fluc)

        # Read in mood sentiment file
        mood_dateframe = pd.read_csv(mood_path)
        mood_vec = mood_dateframe.drop(['date'], axis=1)
        mood_date = mood_dateframe['date']

        # merge process
        # merge stock fluctuation vector and sentiment vector with same date
        merged_vec, merged_label, merged_date = [], [], []

        # i for fluc, j for mood
        i, j = 0, 0
        fluc_size = len(fluc_vec.values)
        mood_size = len(mood_vec.values)
        while(i < fluc_size and j < mood_size):
            if(fluc_date.values.ravel()[i] == mood_date.values[j]):
                merged_vec += [np.append(fluc_vec.values[i], mood_vec.values[j])]
                merged_label += [fluc_y.values[i]]
                merged_date += [fluc_date.values[i]]
                i += 1
                j += 1
            elif(fluc_date.values.ravel()[i] < mood_date.values[j]):
                i += 1
            else:
                j += 1

        # convert list to pandas DataFrame
        merged_vec = {'x'+str(i+1):[item[i] for item in merged_vec] for i in range(0, len(merged_vec[0]))}
        x = pd.DataFrame(merged_vec)

        merged_label = {'y':[item[0] / 100.0 for item in merged_label]}
        y = pd.DataFrame(merged_label)

        merged_date = {'Date':[item[0] for item in merged_date]}
        date = pd.DataFrame(merged_date)

        return x, y, date


def main():
    model = featureExtractor(0)
    x, y, date = model.getFeature(5, 0)
    print(x, y, date)

if __name__ == '__main__':
    main()