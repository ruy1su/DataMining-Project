#!/usr/local/bin/python3

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import tester

def main():
    ts = tester.Tester(5)
    ts.testAll(5, 0)

if __name__ == '__main__':
    main()