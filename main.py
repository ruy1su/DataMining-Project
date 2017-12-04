#!/usr/local/bin/python3

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from utils import tester

def main():
    ts = tester.Tester(5)
    ts.testAll(3, 1)

if __name__ == '__main__':
    main()