#!/usr/local/bin/python3

# -*- coding: utf-8 -*-

import stockParser as sp
import pandas as pd
import numpy as np

def getDataframe(filePath):
    dataframe = pd.read_csv(filePath)
    y = dataframe['y']
    x = dataframe.drop('y', axis=1)
    return x, y

def main():
    x, y = getDataframe(sp.AAPL_PATH)
    print(x)

if __name__ == '__main__':
    main()