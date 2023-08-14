#!/usr/bin/python
from platform import node
import networkx
import networkx.algorithms.community
import sys
import os
import json
import dgl
import torch
import pickle
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import copy
import hashlib
import random
import time
import numpy as np
import tqdm
from exportTrainingSet import CDFG
from joblib import Parallel, delayed

with open(sys.argv[1], 'rb') as cdfg_file:
    cdfg = pickle.load(cdfg_file)

print(cdfg.CFG.number_of_nodes())
print(cdfg.DFG.number_of_nodes())

input("paused")

for i in tqdm.tqdm(range(0, len(cdfg.base_graph.ndata['attr']))):
    L = 2 ** 5
    M = 1
    feature_idx = 3
    if i == 0:
        print("old-->", cdfg.base_graph.ndata['attr'][i])
    while M <= L:
        cdfg.base_graph.ndata['attr'][i][feature_idx] = (2 ** cdfg.base_graph.ndata['attr'][i][feature_idx]) / cdfg.CFG.number_of_nodes() * 100
        feature_idx += 1
        M = M * 2
    
    L = 2 ** 1
    M = 1
    while M <= L:
        cdfg.base_graph.ndata['attr'][i][feature_idx] = (2 ** cdfg.base_graph.ndata['attr'][i][feature_idx]) / cdfg.CFG.number_of_nodes() * 100
        feature_idx += 1
        M = M * 2
    
    L = 2 ** 2
    M = 1
    while M <= L:
        cdfg.base_graph.ndata['attr'][i][feature_idx] = (2 ** cdfg.base_graph.ndata['attr'][i][feature_idx]) / cdfg.DFG.number_of_nodes() * 100
        feature_idx += 1
        M = M * 2
    
    L = 2 ** 1
    M = 1
    while M <= L:
        cdfg.base_graph.ndata['attr'][i][feature_idx] = (2 ** cdfg.base_graph.ndata['attr'][i][feature_idx]) / cdfg.DFG.number_of_nodes() * 100
        feature_idx += 1
        M = M * 2
    if i == 0:
        print("new-->", cdfg.base_graph.ndata['attr'][i])
    if i == 0:
        input("continue?")

with open(sys.argv[2], "wb") as pkf:
    pickle.dump(cdfg, pkf)
