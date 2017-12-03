import numpy as np
import networkx as nx

def ext():
    l = ''
    with open('alertWords.txt') as fin:
        for ls in fin:
            ls = ls.rstrip(',')
            
    print ls
ext()