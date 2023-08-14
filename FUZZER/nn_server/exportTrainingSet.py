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
from joblib import Parallel, delayed

#import pyroscope
#
#pyroscope.configure(
#  application_name = "dcfuzz.fuzzer.exportTrainningSet", # replace this with some name for your application
#  server_address   = "http://172.17.0.1:4040", # replace this with the address of your pyroscope server
#)

data = [1, 2, 3, 4, 5, 6, 7, 8, 20]

bit_idxs_table = [[], [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2], [3], [0, 3], [1, 3], [0, 1, 3], [2, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3], [4], [0, 4], [1, 4], [0, 1, 4], [2, 4], [0, 2, 4], [1, 2, 4], [0, 1, 2, 4], [3, 4], [0, 3, 4], [1, 3, 4], [0, 1, 3, 4], [2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4], [0, 1, 2, 3, 4], [5], [0, 5], [1, 5], [0, 1, 5], [2, 5], [0, 2, 5], [1, 2, 5], [0, 1, 2, 5], [3, 5], [0, 3, 5], [1, 3, 5], [0, 1, 3, 5], [2, 3, 5], [0, 2, 3, 5], [1, 2, 3, 5], [0, 1, 2, 3, 5], [4, 5], [0, 4, 5], [1, 4, 5], [0, 1, 4, 5], [2, 4, 5], [0, 2, 4, 5], [1, 2, 4, 5], [0, 1, 2, 4, 5], [3, 4, 5], [0, 3, 4, 5], [1, 3, 4, 5], [0, 1, 3, 4, 5], [2, 3, 4, 5], [0, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [6], [0, 6], [1, 6], [0, 1, 6], [2, 6], [0, 2, 6], [1, 2, 6], [0, 1, 2, 6], [3, 6], [0, 3, 6], [1, 3, 6], [0, 1, 3, 6], [2, 3, 6], [0, 2, 3, 6], [1, 2, 3, 6], [0, 1, 2, 3, 6], [4, 6], [0, 4, 6], [1, 4, 6], [0, 1, 4, 6], [2, 4, 6], [0, 2, 4, 6], [1, 2, 4, 6], [0, 1, 2, 4, 6], [3, 4, 6], [0, 3, 4, 6], [1, 3, 4, 6], [0, 1, 3, 4, 6], [2, 3, 4, 6], [0, 2, 3, 4, 6], [1, 2, 3, 4, 6], [0, 1, 2, 3, 4, 6], [5, 6], [0, 5, 6], [1, 5, 6], [0, 1, 5, 6], [2, 5, 6], [0, 2, 5, 6], [1, 2, 5, 6], [0, 1, 2, 5, 6], [3, 5, 6], [0, 3, 5, 6], [1, 3, 5, 6], [0, 1, 3, 5, 6], [2, 3, 5, 6], [0, 2, 3, 5, 6], [1, 2, 3, 5, 6], [0, 1, 2, 3, 5, 6], [4, 5, 6], [0, 4, 5, 6], [1, 4, 5, 6], [0, 1, 4, 5, 6], [2, 4, 5, 6], [0, 2, 4, 5, 6], [1, 2, 4, 5, 6], [0, 1, 2, 4, 5, 6], [3, 4, 5, 6], [0, 3, 4, 5, 6], [1, 3, 4, 5, 6], [0, 1, 3, 4, 5, 6], [2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [7], [0, 7], [1, 7], [0, 1, 7], [2, 7], [0, 2, 7], [1, 2, 7], [0, 1, 2, 7], [3, 7], [0, 3, 7], [1, 3, 7], [0, 1, 3, 7], [2, 3, 7], [0, 2, 3, 7], [1, 2, 3, 7], [0, 1, 2, 3, 7], [4, 7], [0, 4, 7], [1, 4, 7], [0, 1, 4, 7], [2, 4, 7], [0, 2, 4, 7], [1, 2, 4, 7], [0, 1, 2, 4, 7], [3, 4, 7], [0, 3, 4, 7], [1, 3, 4, 7], [0, 1, 3, 4, 7], [2, 3, 4, 7], [0, 2, 3, 4, 7], [1, 2, 3, 4, 7], [0, 1, 2, 3, 4, 7], [5, 7], [0, 5, 7], [1, 5, 7], [0, 1, 5, 7], [2, 5, 7], [0, 2, 5, 7], [1, 2, 5, 7], [0, 1, 2, 5, 7], [3, 5, 7], [0, 3, 5, 7], [1, 3, 5, 7], [0, 1, 3, 5, 7], [2, 3, 5, 7], [0, 2, 3, 5, 7], [1, 2, 3, 5, 7], [0, 1, 2, 3, 5, 7], [4, 5, 7], [0, 4, 5, 7], [1, 4, 5, 7], [0, 1, 4, 5, 7], [2, 4, 5, 7], [0, 2, 4, 5, 7], [1, 2, 4, 5, 7], [0, 1, 2, 4, 5, 7], [3, 4, 5, 7], [0, 3, 4, 5, 7], [1, 3, 4, 5, 7], [0, 1, 3, 4, 5, 7], [2, 3, 4, 5, 7], [0, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 7], [0, 1, 2, 3, 4, 5, 7], [6, 7], [0, 6, 7], [1, 6, 7], [0, 1, 6, 7], [2, 6, 7], [0, 2, 6, 7], [1, 2, 6, 7], [0, 1, 2, 6, 7], [3, 6, 7], [0, 3, 6, 7], [1, 3, 6, 7], [0, 1, 3, 6, 7], [2, 3, 6, 7], [0, 2, 3, 6, 7], [1, 2, 3, 6, 7], [0, 1, 2, 3, 6, 7], [4, 6, 7], [0, 4, 6, 7], [1, 4, 6, 7], [0, 1, 4, 6, 7], [2, 4, 6, 7], [0, 2, 4, 6, 7], [1, 2, 4, 6, 7], [0, 1, 2, 4, 6, 7], [3, 4, 6, 7], [0, 3, 4, 6, 7], [1, 3, 4, 6, 7], [0, 1, 3, 4, 6, 7], [2, 3, 4, 6, 7], [0, 2, 3, 4, 6, 7], [1, 2, 3, 4, 6, 7], [0, 1, 2, 3, 4, 6, 7], [5, 6, 7], [0, 5, 6, 7], [1, 5, 6, 7], [0, 1, 5, 6, 7], [2, 5, 6, 7], [0, 2, 5, 6, 7], [1, 2, 5, 6, 7], [0, 1, 2, 5, 6, 7], [3, 5, 6, 7], [0, 3, 5, 6, 7], [1, 3, 5, 6, 7], [0, 1, 3, 5, 6, 7], [2, 3, 5, 6, 7], [0, 2, 3, 5, 6, 7], [1, 2, 3, 5, 6, 7], [0, 1, 2, 3, 5, 6, 7], [4, 5, 6, 7], [0, 4, 5, 6, 7], [1, 4, 5, 6, 7], [0, 1, 4, 5, 6, 7], [2, 4, 5, 6, 7], [0, 2, 4, 5, 6, 7], [1, 2, 4, 5, 6, 7], [0, 1, 2, 4, 5, 6, 7], [3, 4, 5, 6, 7], [0, 3, 4, 5, 6, 7], [1, 3, 4, 5, 6, 7], [0, 1, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]

def combine_map(A:bytes, B:bytes):
    map_array = bytearray(A)
    for i in range(0, len(A)):
        map_array[i] = 0 if (A[i] != 0xff or B[i] != 0xff) else 0xff
    return bytes(map_array)

def count_bits(A:bytes):
    score_A = 0
    for i in range(0, len(A)):
        score_A += 1 if A[i] != 0xff else 0
    return score_A

def get_indexs(X: list):
    idxs = {}
    cnt = 0
    res = []
    for x in X:
        if x not in list(idxs.keys()):
            cnt += 1
            idxs[x] = cnt
        res.append(idxs[x])
    return res

def add_cfg_feature(CFG: networkx.DiGraph, node_idxs, id_bb_dict):
    cfg_features = np.zeros([len(node_idxs), 6])
    for i in range(0, len(node_idxs)):
        node_idx = node_idxs[i]
        node = str(id_bb_dict[node_idx])
        feature_idx = 0
        
        # 2^(1-l)阶可达控制流节点数量级
        L = 2 ** 5
        M = 1
        try:
            all_neighbours = networkx.single_source_shortest_path_length(CFG, node, cutoff = 63).items()
        except:
            all_neighbours = []
        try:
            while M <= L:
                cnt = 0
                cnt = len([k for k, v in all_neighbours if v >= M and v < M * 2])
                if M == 1:
                    cnt += 1
                cfg_features[i][feature_idx] = float(cnt + 1) / CFG.number_of_nodes() * 100
                feature_idx += 1
                M = M * 2
        except:
            continue
    return node_idxs, cfg_features

def get_boundary_nodes(G:networkx.DiGraph, covered_nodes:set, node_idx_name_table, node_name_idx_table, bb_id_dict):
    nei_nodes = set()
    #adj_matrix = np.array(networkx.adjacency_matrix(G.copy()).todense(), dtype=np.uint8)

    #adj_matrix_g1 = np.zeros(adj_matrix.shape, dtype=np.uint8)
    covered_node_names = [str(x) for x in node_idx_name_table[np.array(list(covered_nodes), dtype=np.uint32)]]
    reshaped_g1 = G.copy().subgraph(covered_node_names)
    #g1_node_idx_name_table = np.array(reshaped_g1.nodes(), dtype=np.uint32)
    #adj_matrix_reshaped_g1 = np.array(networkx.adjacency_matrix(reshaped_g1).todense(), dtype=np.uint8)
    #adj_matrix_g1[node_name_idx_table[g1_node_idx_name_table[np.arange(0, len(g1_node_idx_name_table), 1)]]] = adj_matrix_reshaped_g1[np.arange(0, len(g1_node_idx_name_table), 1)]
    #adj_matrix_g1[:, node_name_idx_table[g1_node_idx_name_table[np.arange(0, len(g1_node_idx_name_table), 1)]]] = adj_matrix_reshaped_g1[:, np.arange(0, len(g1_node_idx_name_table), 1)]

    #adj_matrix_g2 = np.zeros(adj_matrix.shape, dtype=np.uint8)
    reshaped_g2 = G.copy()
    reshaped_g2.remove_nodes_from(covered_node_names)
    #g2_node_idx_name_table = np.array(reshaped_g2.nodes(), dtype=np.uint32)
    #adj_matrix_reshaped_g2 = np.array(networkx.adjacency_matrix(reshaped_g2).todense(), dtype=np.uint8)
    #adj_matrix_g2[node_name_idx_table[g2_node_idx_name_table[np.arange(0, len(g2_node_idx_name_table), 1)]]] = adj_matrix_reshaped_g2[np.arange(0, len(g2_node_idx_name_table), 1)]
    #adj_matrix_g2[:, node_name_idx_table[g2_node_idx_name_table[np.arange(0, len(g2_node_idx_name_table), 1)]]] = adj_matrix_reshaped_g2[:, np.arange(0, len(g2_node_idx_name_table), 1)]

    edges = set(G.edges()) - set(reshaped_g1.edges()) - set(reshaped_g2.edges())
    #boundary_matrix = adj_matrix - adj_matrix_g2
    #boundary_matrix -= adj_matrix_g1
    #edges = np.nonzero(boundary_matrix)
    for edge in edges:
        nei_node_idx = bb_id_dict[edge[1]]
        if nei_node_idx not in covered_nodes and nei_node_idx not in nei_nodes:
            nei_nodes.add(nei_node_idx)
    #print(nei_nodes)
    return nei_nodes

class CDFG:
    def __init__(self, need_split = False):
        #DG
        self.data_nodes = set()
        self.data_nodes_type = {}
        self.data_edges = set()
        #CFG
        self.cfg_nodes = set()
        self.cfg_nodes_feature = {}
        self.cfg_edges = set()
        #DCG
        self.dcg_edges = set()
        #CG temporary
        self.cg_edges = set()
        self.need_split = need_split

        self.uncovered_nodes = set()
        self.covered_nodes = set()
        #self.uncovered_edges = set()
        #self.covered_edges = set()

    def __load_cfg(self, path):
        f = open(path, "r")
        try:
            f_feature = open(path[:path.rfind(".cfg")] + ".bbf")
        except:
            f.close()
            return
        feature = {}
        for line in f_feature.readlines():
            info = line.split(":")
            feature[info[2].strip()] = [int(x.strip()) for x in info[3:]]
        lines = f.readlines()
        if self.need_split and len(lines) > 30:
            if len(lines) % 3 == 0:
                A = [x.strip().split(":")[-1] for x in lines[0: len(lines) // 3]]
                B = [x.strip().split(":")[-1] for x in lines[len(lines) // 3 * 1: len(lines) // 3 * 2]]
                C = [x.strip().split(":")[-1] for x in lines[len(lines) // 3 * 2: ]]
                if get_indexs(A) == get_indexs(B) and get_indexs(B) == get_indexs(C):
                    lines = lines[len(lines) // 3 * 2: ]
            elif len(lines) % 2 == 0:
                A = [x.strip().split(":")[-1] for x in lines[0: len(lines) // 2]]
                B = [x.strip().split(":")[-1] for x in lines[len(lines) // 2:]]
                if get_indexs(A) == get_indexs(B):
                    lines = lines[len(lines) // 2:]
        for line in lines:
            info = line.split(":")
            if "/" in info[0]:
                info[0] = info[0][info[0].find("/"): ]
            info[0] = info[0].replace("/", "_")
            #cdf.add_cfg_edge(info[2].strip(), info[3].strip(), info[1])
            src = info[2].strip()
            dst = info[3].strip()
            func = info[1]
            if src not in self.cfg_nodes:
                self.cfg_nodes.add(src)
                self.cfg_nodes_feature[src] = feature[src]
            if dst not in self.cfg_nodes:
                self.cfg_nodes.add(dst)
                self.cfg_nodes_feature[dst] = feature[dst]
            self.cfg_edges.add((src, dst, func))
        f.close()
        f_feature.close()
    
    def __load_dg(self, path):
        f = open(path, "r")
        conts = f.read()
        if "}\n{" in conts:
            cont = "{" + conts.split("}\n{")[-1]
        else:
            cont = conts
        infos = json.loads(cont)
        try:
            f_feature = open(path[:path.rfind(".dg")] + ".bbf")
        except:
            f.close()
            return
        feature = {}
        for line in f_feature.readlines():
            info = line.split(":")
            feature[info[2].strip()] = [int(x.strip()) for x in info[3:]]
        for info in infos["dataflow"]:
            try:
                #srcs = [x for x in info["rvar"] if not x.startswith("#")]
                srcs = [x for x in info["rvar"]]
                dst = info["lvar"]
                bb = str(int(info["BB"][4:].split("_")[0]))
                for src in srcs:
                    #if dst.startswith("#"): # not static
                    #    continue
                    if src["Name"] not in self.data_nodes:
                        self.data_nodes.add(src["Name"])
                        self.data_nodes_type[src["Name"]] = src["Type"]
                    if dst["Name"] not in self.data_nodes:
                        self.data_nodes.add(dst["Name"])
                        self.data_nodes_type[dst["Name"]] = dst["Type"]
                    self.data_edges.add((src["Name"], dst["Name"], bb, info["OPCode"]))
            except:
                continue
        f.close()
        f_feature.close()
    
    def __load_dcg(self, path):
        f = open(path, "r")
        try:
            f_feature = open(path[:path.rfind(".dcg")] + ".bbf")
        except:
            f.close()
            return
        feature = {}
        for line in f_feature.readlines():
            info = line.split(":")
            feature[info[2].strip()] = [int(x.strip()) for x in info[3:]]
        conts = f.read()
        if "}\n{" in conts:
            cont = "{" + conts.split("}\n{")[-1]
        else:
            cont = conts
        infos = json.loads(cont)
        for info in infos["dcgflow"]:
            for _dst in info["BBdst"]:
                src = str(int(info["BBsrc"][4:].split("_")[0]))
                dst = str(int(_dst[4:].split("_")[0]))
                var = info["BBvar"]
                if var not in self.data_nodes:
                    #print(var)
                    self.data_nodes.add(var)
                self.dcg_edges.add((var, dst, var))
        f_feature.close()
        f.close()

    def __load_cg(self, path):
        f = open(path, "r")
        try:
            f_feature = open(path[:path.rfind(".cg")] + ".bbf")
        except:
            f.close()
            return
        feature = {}
        for line in f_feature.readlines():
            info = line.split(":")
            feature[info[2].strip()] = [int(x.strip()) for x in info[3:]]
        lines = f.readlines()
        if self.need_split and len(lines) > 30:
            if len(lines) % 3 == 0:
                A = [x.strip().split(":")[3] for x in lines[0: len(lines) // 3]]
                B = [x.strip().split(":")[3] for x in lines[len(lines) // 3 * 1: len(lines) // 3 * 2]]
                C = [x.strip().split(":")[3] for x in lines[len(lines) // 3 * 2: ]]
                if get_indexs(A) == get_indexs(B) and get_indexs(B) == get_indexs(C):
                    lines = lines[len(lines) // 3 * 2: ]
            elif len(lines) % 2 == 0:
                A = [x.strip().split(":")[3] for x in lines[0: len(lines) // 2]]
                B = [x.strip().split(":")[3] for x in lines[len(lines) // 2:]]
                if get_indexs(A) == get_indexs(B):
                    lines = lines[len(lines) // 2:]
        for line in lines:
            info = line.strip().split(":")
            info[0] = info[0].replace("/", "_")
            src = info[1].strip()
            dst = info[2].strip()
            src_bb = info[3]
            lvar = info[4]
            args = info[5:]
            # HANDLE CFG
            for cfg_eg in self.cfg_edges:
                if cfg_eg[2] == dst:
                    self.cfg_edges.add((src_bb, cfg_eg[0], src))
                    if src_bb not in self.cfg_nodes:
                        self.cfg_nodes.add(src_bb)
                        try:
                            self.cfg_nodes_feature[src_bb] = feature[src_bb]
                        except:
                            self.cfg_nodes_feature[src_bb] = [0,0,0]
                    if cfg_eg[0] not in self.cfg_nodes:
                        self.cfg_nodes.add(cfg_eg[0])
                        if cfg_eg[0] not in feature.keys():
                            self.cfg_nodes_feature[cfg_eg[0]] = self.__search_feature(cfg_eg[0])
                        else:
                            self.cfg_nodes_feature[cfg_eg[0]] = feature[cfg_eg[0]]
                    break
            # HANDLE DG
            for argidx in range(0, len(args)):
                arg = args[argidx]
                #if arg.startswith("#"): # not static
                #    continue
                if dst + "%" + str(argidx + 1) in self.data_nodes:
                    dg_nd = dst + "%" + str(argidx + 1)
                    if arg not in self.data_nodes:
                        self.data_nodes.add(arg)
                    if dg_nd not in self.data_nodes:
                        self.data_nodes.add(dg_nd)
                    self.data_edges.add((arg, dg_nd, src_bb))
                    break
            self.cg_edges.add((src, dst, src_bb, lvar))
        f_feature.close()
        f.close()
    
    def __search_feature(self, bname):
        for dirpath, dirnames, filenames in os.walk(CDF_FILE_DIR):
            for filename in filenames:
                if filename.endswith(".bbf"):
                    _ff = open(os.path.join(dirpath, filename))
                    for line in _ff.readlines():
                        info = line.split(":")
                        if info[2].strip() == bname:
                            _ff.close()
                            return [int(x.strip()) for x in info[3:]]
                    _ff.close()
        raise FileNotFoundError

    def __load_rg(self, path):
        f = open(path, "r")
        f_feature = open(path[:path.rfind(".rg")] + ".bbf")
        feature = {}
        for line in f_feature.readlines():
            info = line.split(":")
            feature[info[2].strip()] = [int(x.strip()) for x in info[3:]]
        lines = f.readlines()
        if self.need_split and len(lines) > 30:
            if len(lines) % 3 == 0:
                A = [x.strip().split(":")[2] for x in lines[0: len(lines) // 3]]
                B = [x.strip().split(":")[2] for x in lines[len(lines) // 3 * 1: len(lines) // 3 * 2]]
                C = [x.strip().split(":")[2] for x in lines[len(lines) // 3 * 2: ]]
                if get_indexs(A) == get_indexs(B) and get_indexs(B) == get_indexs(C):
                    lines = lines[len(lines) // 3 * 2: ]
            elif len(lines) % 2 == 0:
                A = [x.strip().split(":")[2] for x in lines[0: len(lines) // 2]]
                B = [x.strip().split(":")[2] for x in lines[len(lines) // 2:]]
                if get_indexs(A) == get_indexs(B):
                    lines = lines[len(lines) // 2:]
        for line in lines:
            info = line.strip().split(":")
            if "/" in info[0]:
                info[0] = info[0][info[0].find("/"): ]
            info[0] = info[0].replace("/", "_")
            src = info[1]
            src_bb = info[2]
            rvar = info[3]
            for cg_edge in self.cg_edges:
                if cg_edge[1] == src:
                    # HANDLE CFG
                    self.cfg_edges.add((src_bb, cg_edge[2], src))
                    if src_bb not in self.cfg_nodes:
                        self.cfg_nodes.add(src_bb)
                        try:
                            self.cfg_nodes_feature[src_bb] = feature[src_bb]
                        except:
                            self.cfg_nodes_feature[src_bb] = [0,0,0]
                    if cg_edge[2] not in self.cfg_nodes:
                        self.cfg_nodes.add(cg_edge[2])
                        if cg_edge[2] not in feature.keys():
                            try:
                                self.cfg_nodes_feature[cg_edge[2]] = self.__search_feature(cg_edge[2])
                            except:
                                continue
                        else:    
                            self.cfg_nodes_feature[cg_edge[2]] = feature[cg_edge[2]]
                    #if rvar.startswith("#"):
                    #    continue
                    if cg_edge[3].startswith("#"):
                        continue
                    if rvar not in self.data_nodes:
                        self.data_nodes.add(rvar)
                    if cg_edge[2] not in self.cfg_nodes:
                        self.data_nodes.add(cg_edge[2].strip())
                    if cg_edge[3] not in self.data_nodes:
                        self.data_nodes.add(cg_edge[3])
                    self.data_edges.add((rvar, cg_edge[3], src_bb))
        f.close()

    def loadCDFG(self, CDF_FILE_DIR):
        #LOCAL
        for dirpath, dirnames, filenames in os.walk(CDF_FILE_DIR):
            for filename in filenames:
                if filename.endswith(".cfg"):
                    print(filename)
                    self.__load_cfg(os.path.join(dirpath, filename))
        #DG
        for dirpath, dirnames, filenames in os.walk(CDF_FILE_DIR):
            for filename in filenames:
                if filename.endswith(".dg"):
                    print(filename)
                    self.__load_dg(os.path.join(dirpath, filename))
        #DCG
        for dirpath, dirnames, filenames in os.walk(CDF_FILE_DIR):
            for filename in filenames:
                if filename.endswith(".dcg"):
                    print(filename)
                    self.__load_dcg(os.path.join(dirpath, filename))
        #CG
        for dirpath, dirnames, filenames in os.walk(CDF_FILE_DIR):
            for filename in filenames:
                if filename.endswith(".cg"):
                    print(filename)
                    self.__load_cg(os.path.join(dirpath, filename))
        #RG
        for dirpath, dirnames, filenames in os.walk(CDF_FILE_DIR):
            for filename in filenames:
                if filename.endswith(".rg"):
                    print(filename)
                    self.__load_rg(os.path.join(dirpath, filename))
    
    def exportCDFG(self, gexf_path):
        G = networkx.DiGraph()
        for node in self.cfg_nodes:
            G.add_node(node, viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
        for edge in self.cfg_edges:
            G.add_edge(edge[0], edge[1], viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
        for node in self.data_nodes:
            G.add_node(node, viz = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 1.0}})
        for edge in self.data_edges:
            G.add_edge(edge[0], edge[1], viz = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 1.0}})
            G.add_edge(edge[2], edge[0], viz = {'color': {'r': 255, 'g': 0, 'b': 255, 'a': 1.0}})
        for edge in self.dcg_edges:
            G.add_edge(edge[2], edge[1], viz = {'color': {'r': 0, 'g': 255, 'b': 0, 'a': 1.0}})
        networkx.write_gexf(G, gexf_path)

    def exportGlobalCFG(self, gexf_path):
        G = networkx.DiGraph()
        for node in self.cfg_nodes:
            G.add_node(node, viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
        for edge in self.cfg_edges:
            G.add_edge(edge[0], edge[1], viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
        networkx.write_gexf(G, gexf_path)
        
    def preprocessGraph(self):
        self.CFG = networkx.DiGraph()
        self.DFG = networkx.DiGraph()
        self.CDG = networkx.DiGraph()
        self.DCG = networkx.DiGraph()

        #DATA IDS
        id_cnt = 0
        self.id_dict = {}
        for node in self.data_nodes:
            self.id_dict[node] = id_cnt
            id_cnt += 1
        #BB IDS
        id_cnt = 0
        self.bb_id_dict = {}
        for node in self.cfg_nodes:
            self.bb_id_dict[node] = id_cnt
            id_cnt += 1

        #self.CFG
        for node in tqdm.tqdm(self.cfg_nodes):
            self.CFG.add_node(node, viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
        for edge in tqdm.tqdm(self.cfg_edges):
            self.CFG.add_edge(edge[0], edge[1], viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
        #self.DFG
        for node in tqdm.tqdm(self.data_nodes):
            self.DFG.add_node(node, viz = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 1.0}})
        for edge in tqdm.tqdm(self.data_edges):
            self.DFG.add_edge(edge[0], edge[1], viz = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 1.0}})
        #self.CDG & self.DCG
        for node in tqdm.tqdm(self.cfg_nodes):
            self.CDG.add_node(node, viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
            self.DCG.add_node(node, viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
        for node in tqdm.tqdm(self.data_nodes):
            self.CDG.add_node(node, viz = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 1.0}})
            self.DCG.add_node(node, viz = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 1.0}})
        for edge in tqdm.tqdm(self.data_edges):
            self.CDG.add_edge(edge[2], edge[0], viz = {'color': {'r': 255, 'g': 0, 'b': 255, 'a': 1.0}})
        for edge in tqdm.tqdm(self.dcg_edges):
            self.DCG.add_edge(edge[2], edge[1], viz = {'color': {'r': 0, 'g': 255, 'b': 0, 'a': 1.0}})
        
        # EXTEND
        for edge in tqdm.tqdm(self.CDG.edges):
            if edge[0] not in self.cfg_nodes:
                self.cfg_nodes.add(edge[0])
                self.bb_id_dict[edge[0]] = id_cnt
                id_cnt += 1
                self.cfg_edges.add((edge[0], edge[0], "NULL"))
                try:
                    self.cfg_nodes_feature[edge[0]] = self.__search_feature(edge[0])
                except:
                    self.cfg_nodes_feature[edge[0]] = [0,0,0]
                self.CFG.add_node(edge[0], viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
                self.CFG.add_edge(edge[0], edge[0], viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})

        for edge in tqdm.tqdm(self.DCG.edges):
            if edge[1] not in self.cfg_nodes:
                self.cfg_nodes.add(edge[1])
                self.bb_id_dict[edge[1]] = id_cnt
                id_cnt += 1
                self.cfg_edges.add((edge[1], edge[1], "NULL"))
                try:
                    self.cfg_nodes_feature[edge[1]] = self.__search_feature(edge[1])
                except:
                    self.cfg_nodes_feature[edge[1]] = [0,0,0]
                self.CFG.add_node(edge[1], viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
                self.CFG.add_edge(edge[1], edge[1], viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
        
        for edge in tqdm.tqdm(self.data_edges):
            if edge[2] not in self.cfg_nodes:
                self.cfg_nodes.add(edge[2])
                self.bb_id_dict[edge[2]] = id_cnt
                id_cnt += 1
                self.cfg_edges.add((edge[2], edge[2], "NULL"))
                try:
                    self.cfg_nodes_feature[edge[2]] = self.__search_feature(edge[2])
                except:
                    self.cfg_nodes_feature[edge[2]] = [0,0,0]
                self.CFG.add_node(edge[2], viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})
                self.CFG.add_edge(edge[2], edge[2], viz = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}})

        # CONSTRUCT DGL HETEROGRAPH
        # graph_data = {
        #     ('BB', 'goto', 'BB'): [(bb_id_dict[edge[0]], bb_id_dict[edge[1]]) for edge in self.CFG.edges],
        #     ('BB', 'trigger', 'DATA'): [(bb_id_dict[edge[0]], id_dict[edge[1]]) for edge in self.CDG.edges],
        #     ('DATA', 'affect', 'DATA'): [(id_dict[edge[0]], id_dict[edge[1]]) for edge in self.DFG.edges],
        #     ('DATA', 'determine', 'BB'): [(id_dict[edge[0]], bb_id_dict[edge[1]]) for edge in self.DCG.edges]
        # }
        self.base_graph = dgl.graph(
            (
                [self.bb_id_dict[edge[0]] for edge in self.CFG.edges],
                [self.bb_id_dict[edge[1]] for edge in self.CFG.edges]
            )
        )
        self.base_graph.add_edges([max(list(self.bb_id_dict.values())), ], [max(list(self.bb_id_dict.values())), ])
        print(self.base_graph)

        self.id_bb_dict = [0 for i in range(0, max(list(self.bb_id_dict.values())) + 1)]
        for key,value in self.bb_id_dict.items():
            self.id_bb_dict[value] = int(key)

        # ADD FEATURE
        nodes_features = np.zeros([len(self.base_graph.nodes()), 19], dtype=np.float32)

        for node_idx in tqdm.tqdm(range(0, len(self.base_graph.nodes()))):
            node = str(self.id_bb_dict[node_idx])
            feature_idx = 0
            try:
                for feat in self.cfg_nodes_feature[node]:
                    nodes_features[node_idx][feature_idx] = feat
                    feature_idx += 1
            except:
                continue


        with Parallel(n_jobs=16, prefer="processes", backend="loky") as parallel:
            results = parallel(
                delayed(add_cfg_feature)(
                    self.CFG,
                    list(range(node_idx, node_idx+100))
                    if node_idx+100 < len(self.base_graph.nodes())
                    else list(range(node_idx, len(self.base_graph.nodes()))),
                    self.id_bb_dict
                ) for node_idx in tqdm.tqdm(range(0, len(self.base_graph.nodes()), 100))
            )
            for result in results:
                idxs, feats = result
                #print(result[1])
                nodes_features[idxs[0]:idxs[-1]+1, len(self.cfg_nodes_feature[node]):len(self.cfg_nodes_feature[node])+6] += feats

        for node_idx in tqdm.tqdm(range(0, len(self.base_graph.nodes()))):
            feature_idx = len(self.cfg_nodes_feature[node]) + 6
            # 2^(1-m)阶可影响控制流节点数量级
            L = 2 ** 1
            M = 1
            feature_idx_bak = feature_idx
            try:
                while M <= L:
                    cnt = 0
                    k_neighbours = networkx.single_source_shortest_path_length(self.CDG, node, cutoff = 1)
                    Nk = [n for (n, v) in k_neighbours.items() if v == 1]
                    for data in Nk:
                        k_neighbours_k_d = networkx.single_source_shortest_path_length(self.DFG, data, cutoff = M * 2)
                        for k in range(M, M * 2):
                            Nkd = [n for (n, v) in k_neighbours_k_d.items() if v == k-1]
                            for _data in Nkd:
                                k_neighbours_1_b = networkx.single_source_shortest_path_length(self.DCG, _data, cutoff = 1)
                                Nkb = [n for (n, v) in k_neighbours_1_b.items() if v == 1]
                                cnt += len(Nkb)
                    nodes_features[node_idx][feature_idx] = float(cnt + 1) / self.CFG.number_of_nodes() * 100
                    feature_idx += 1
                    M = M * 2
            except Exception as e:
                feature_idx = feature_idx_bak + 2

            # 2^(1-n)阶可影响数据流节点数量级
            L = 2 ** 2
            M = 1
            feature_idx_bak = feature_idx
            try:
                while M <= L:
                    cnt = 0
                    k_neighbours = networkx.single_source_shortest_path_length(self.CDG, node, cutoff = 1)
                    Nk = [n for (n, v) in k_neighbours.items() if v == 1]
                    for data in Nk:
                        k_neighbours_k_d = networkx.single_source_shortest_path_length(self.DFG, data, cutoff = M * 2)
                        for k in range(M, M * 2):
                            Nkd = [n for (n, v) in k_neighbours_k_d.items() if v == k-1]
                            cnt += len(Nkd)
                    nodes_features[node_idx][feature_idx] = float(cnt + 1) / self.DFG.number_of_nodes() * 100
                    feature_idx += 1
                    M = M * 2
            except Exception as e:
                feature_idx = feature_idx_bak + 3

            # 2^(1-o)阶受影响数据流节点数量级
            L = 2 ** 1
            M = 1
            feature_idx_bak = feature_idx
            try:
                condition_data = networkx.single_target_shortest_path_length(self.DCG, node, cutoff = 1)
                condition_data = [n for (n, v) in condition_data if v == 1]
                while M <= L:
                    cnt = 0
                    for data_node in condition_data:
                        k_prev_neighbours = networkx.single_target_shortest_path_length(self.DFG, data_node, cutoff = M * 2)
                        for k in range(M, M * 2):
                            Nk = [n for (n, v) in list(k_prev_neighbours) if v == k-1]
                            cnt += len(Nk)
                    nodes_features[node_idx][feature_idx] = float(cnt + 1) / self.DFG.number_of_nodes() * 100
                    feature_idx += 1
                    M = M * 2
            except:
                feature_idx = feature_idx_bak + 2
        
        #for i in range(0, len(nodes_features[0])):
        #    drawBoxPlot(nodes_features[x][i] for x in range(0, len(nodes_features)))

        tensor_nodes_features = torch.tensor(nodes_features, dtype=torch.float32)
        print(len(tensor_nodes_features))
        self.base_graph.ndata["attr"] = tensor_nodes_features
        
        self.uncovered_nodes = set([i for i in range(0, len(self.id_bb_dict))])
        self.covered_nodes = set()
        #self.uncovered_edges = set([i for i in range(0, len(self.base_graph.edges))])
        #self.covered_edges = set()
        edges = self.base_graph.edges(form = 'uv')
        self.hash_edge = {}
        for i in range(0, len(edges[0])):
            src = self.id_bb_dict[edges[0][i].item()]
            dst = self.id_bb_dict[edges[1][i].item()]
            hash_val = (src >> 1) ^ dst
            self.hash_edge[hash_val] = (edges[0][i].item(), edges[1][i].item())

    def exportTrainingSet(self, trainning_set_dir: str, export_dir: str):

        all_seed = {}
        corpus_dir = os.path.join(trainning_set_dir, "corpus")
        seed_dir = os.path.join(trainning_set_dir, "output")
        for seed in os.listdir(seed_dir):
            seed_hash = seed.split(":")[-1].split(".")[0]
            all_seed[seed_hash] = seed
        selected = [0 for i in range(0, 1000)]
        random.seed(time.time())
        root = corpus_dir
        for i in range(2, 1000):
            for j in range(0, 1):
                _dir = "{}_{}".format(i, random.randrange(1, 101))
                corpus_fd = open(os.path.join(root, _dir, "corpus"), "r")
                first_virgin_combine = None
                
                for line in corpus_fd.readlines():
                    seed_name = line.strip()
                    seed_hash = seed_name.split(":")[-1].split(".")[0]
                    if first_virgin_combine == None:
                        with open(os.path.join(seed_dir, all_seed[seed_hash], "first_virgin"), "rb") as fvd:
                            first_virgin_combine = fvd.read()
                    else:
                        with open(os.path.join(seed_dir, all_seed[seed_hash], "first_virgin"), "rb") as fvd:
                            first_virgin = fvd.read()
                        combine_map(first_virgin_combine, first_virgin)

                bits_cnt_total = count_bits(first_virgin_combine)
                corpus_fd.seek(0)

                for line in corpus_fd.readlines():
                    seed_name = line.strip()
                    seed_hash = seed_name.split(":")[-1].split(".")[0]
                    with open(os.path.join(seed_dir, all_seed[seed_hash], "last_virgin"), "rb") as lvd:
                        last_virgin = lvd.read()
                    tag = count_bits(combine_map(first_virgin_combine, last_virgin)) / bits_cnt_total

                    dgl_export = self.base_graph.clone()
                    edges =  dgl_export.edges(form = 'uv')

                    for i in range(0, len(edges[0])):
                        src = self.id_bb_dict[edges[0][i].item()]
                        dst = self.id_bb_dict[edges[1][i].item()]
                        hash_val = (src >> 1) ^ dst
                        if first_virgin[hash_val] != 0xff:
                            dgl_export.ndata['attr'][edges[0][i].item()][-3] = 1
                            dgl_export.ndata['attr'][edges[1][i].item()][-3] = 1
                        elif first_virgin_combine[hash_val] != 0xff:
                            dgl_export.ndata['attr'][edges[0][i].item()][-2] = 1
                            dgl_export.ndata['attr'][edges[1][i].item()][-2] = 1

                    for node in range(0, dgl_export.num_nodes()):
                        if dgl_export.ndata['attr'][node][-2] == 1:
                            for nei_node in networkx.single_source_shortest_path_length(self.CFG, str(self.id_bb_dict[node]), cutoff = 2):
                                nei_node_idx = self.bb_id_dict[nei_node]
                                if dgl_export.ndata['attr'][nei_node_idx][-2] == 0 and dgl_export.ndata['attr'][nei_node_idx][-3] == 0:
                                    dgl_export.ndata['attr'][nei_node_idx][-1] = 1
                    
                    masked_nodes = []
                    for node in range(0, dgl_export.num_nodes()):
                        if dgl_export.ndata['attr'][node][-3] == 0 and dgl_export.ndata['attr'][node][-2] == 0 and dgl_export.ndata['attr'][node][-1] == 0:
                            masked_nodes.append(node)
                    dgl_export.remove_nodes(torch.tensor(masked_nodes))

                    print(dgl_export)
                    dgl.save_graphs(os.path.join(export_dir, seed_hash + "_" + hashlib.sha256(first_virgin_combine).hexdigest() + "_" + str(tag) + ".dgl"), [dgl_export, ])
                
                corpus_fd.close()


        #model = TSNE(n_components=3)
        #node_pos = model.fit_transform(tensor_nodes_features)
        #ax = plt.axes(projection ="3d")
        #X = [node_pos[idx, 0] for idx in range(0, len(node_pos))]
        #Y = [node_pos[idx, 1] for idx in range(0, len(node_pos))]
        #Z = [node_pos[idx, 2] for idx in range(0, len(node_pos))]
        #ax.scatter3D(X, Y, Z, alpha = 0.8)  # c=node_colors)
        #ax.axes.set_xlim3d(left=sorted(X)[int(len(X) * 0.05)], right=sorted(X)[int(len(X) * 0.95)])
        #ax.axes.set_ylim3d(bottom=sorted(Y)[int(len(Y) * 0.05)], top=sorted(Y)[int(len(X) * 0.95)])
        #ax.axes.set_zlim3d(bottom=sorted(Z)[int(len(Z) * 0.05)], top=sorted(Z)[int(len(X) * 0.95)])
        #plt.legend()
        #plt.show()


    def exportGraph(self, virgin_bits, seed_bits, virgin_changed=True):
        if max(list(self.bb_id_dict.values())) not in self.base_graph.nodes():
            self.base_graph.add_edges([max(list(self.bb_id_dict.values())), ], [max(list(self.bb_id_dict.values())), ])
        if virgin_changed:
            try:
                self.nei_nodes
            except:
                self.nei_nodes = set()
            new_nei_nodes = set()
            self.dgl_virgin_base = self.base_graph.clone()
            edges = self.dgl_virgin_base.edges(form = 'uv')
            ndatas = self.dgl_virgin_base.ndata['attr'].numpy()
            for i in range(0, len(edges[0])):
                src = self.id_bb_dict[edges[0][i].item()]
                dst = self.id_bb_dict[edges[1][i].item()]
                hash_val = (src >> 1) ^ dst
                if virgin_bits[hash_val] != 0xff:
                    ndatas[edges[0][i].item()][-2] = 1
                    ndatas[edges[1][i].item()][-2] = 1
                    if edges[0][i].item() not in self.covered_nodes:
                        self.covered_nodes.add(edges[0][i].item())
                        self.uncovered_nodes.remove(edges[0][i].item())
                        if edges[0][i].item() in self.nei_nodes:
                            self.nei_nodes.remove(edges[0][i].item())
                            if edges[0][i].item() in new_nei_nodes:
                                new_nei_nodes.remove(edges[0][i].item())
                            for node, dist in dict(networkx.single_source_shortest_path_length(self.CFG, str(src), cutoff=5)).items():
                                if dist > 0:
                                    if (self.bb_id_dict[node] not in self.covered_nodes) and (self.bb_id_dict[node] not in self.nei_nodes):
                                        self.nei_nodes.add(self.bb_id_dict[node])
                                        new_nei_nodes.add(self.bb_id_dict[node])
                        elif len(self.nei_nodes) > 0:
                            for node, dist in dict(networkx.single_source_shortest_path_length(self.CFG, str(src), cutoff=5)).items():
                                if dist > 0:
                                    if (self.bb_id_dict[node] not in self.covered_nodes) and (self.bb_id_dict[node] not in self.nei_nodes):
                                        self.nei_nodes.add(self.bb_id_dict[node])
                                        new_nei_nodes.add(self.bb_id_dict[node])
                    if edges[1][i].item() not in self.covered_nodes:
                        self.covered_nodes.add(edges[1][i].item())
                        self.uncovered_nodes.remove(edges[1][i].item())
                        if edges[1][i].item() in self.nei_nodes:
                            self.nei_nodes.remove(edges[1][i].item())
                            if edges[1][i].item() in new_nei_nodes:
                                new_nei_nodes.remove(edges[1][i].item())
                            for node, dist in dict(networkx.single_source_shortest_path_length(self.CFG, str(dst), cutoff=5)).items():
                                if dist > 0:
                                    if (self.bb_id_dict[node] not in self.covered_nodes) and (self.bb_id_dict[node] not in self.nei_nodes):
                                        self.nei_nodes.add(self.bb_id_dict[node])
                                        new_nei_nodes.add(self.bb_id_dict[node])
                        elif len(self.nei_nodes) > 0:
                            for node, dist in dict(networkx.single_source_shortest_path_length(self.CFG, str(dst), cutoff=5)).items():
                                if dist > 0:
                                    if (self.bb_id_dict[node] not in self.covered_nodes) and (self.bb_id_dict[node] not in self.nei_nodes):
                                        self.nei_nodes.add(self.bb_id_dict[node])
                                        new_nei_nodes.add(self.bb_id_dict[node])

            node_idx_name_table = np.array(self.CFG.nodes(), dtype=np.uint32)
            node_name_idx_table = np.zeros(max(node_idx_name_table)+1, dtype=np.uint32)
            node_name_idx_table[node_idx_name_table] = np.arange(0, len(node_idx_name_table), 1)

            if len(self.nei_nodes) == 0:
                for STEP in range(0, 5):
                    next_nei_nodes = get_boundary_nodes(self.CFG, self.covered_nodes | self.nei_nodes, node_idx_name_table, node_name_idx_table, self.bb_id_dict)
                    for nei_node_idx in next_nei_nodes:
                        if nei_node_idx not in self.covered_nodes or (ndatas[nei_node_idx][-2] == 0 and ndatas[nei_node_idx][-3] == 0):
                            ndatas[nei_node_idx][-1] = 1
                    self.nei_nodes |= next_nei_nodes
            else:
                for nei_node_idx in new_nei_nodes:
                    ndatas[nei_node_idx][-1] = 1

                
            self.dgl_virgin_base.ndata['attr'] = torch.tensor(ndatas)
            self.masked_nodes = list(self.uncovered_nodes - self.nei_nodes)

            # [0, 1, 2, 3, 4] --> [0, 1, 2(3), 3(4)]
            self.node2oldnode_tbl = np.arange(self.base_graph.num_nodes())
            for i in range(0, len(self.masked_nodes)):
                self.node2oldnode_tbl[self.masked_nodes[i]+1: ] -= 1

            self.dgl_virgin_base.remove_nodes(torch.tensor(self.masked_nodes))
            
        dgl_export = self.dgl_virgin_base.clone()

        ndatas = dgl_export.ndata['attr'].numpy()

        seed_bits_u8_array = np.frombuffer(seed_bits, dtype=np.uint8)

        for idx in np.nonzero(seed_bits_u8_array)[0]:
            for offset in bit_idxs_table[seed_bits[idx]]:
                hash_val = (idx << 3) | offset
                src, dst = self.hash_edge.get(hash_val, (None, None))

                if src == None:
                    continue
                
                ndatas[self.node2oldnode_tbl[src]][-2] = 0
                ndatas[self.node2oldnode_tbl[dst]][-2] = 0

                ndatas[self.node2oldnode_tbl[src]][-3] = 1
                ndatas[self.node2oldnode_tbl[dst]][-3] = 1
        
        dgl_export.ndata['attr'] = torch.tensor(ndatas)
                        
        #dgl_export.remove_nodes(torch.tensor(self.masked_nodes))
        return dgl_export

#def drawBoxPlot(data):
#    df = pd.DataFrame(data)
#    print(df.describe())
#    df.plot.box(title="Box Chart")
#    plt.grid(linestyle="--", alpha=0.3)
#    plt.show()

    
if __name__ == "__main__":
    CDF_FILE_DIR = sys.argv[1]
    if os.access(os.path.join(sys.argv[2], "CDFG.pkl"), os.R_OK):
        with open(os.path.join(sys.argv[2], "CDFG.pkl"), "rb") as pkf:
            cdfg = pickle.load(pkf)
    else:
        cdfg = CDFG(need_split=True)
        cdfg.loadCDFG(CDF_FILE_DIR)
        cdfg.preprocessGraph()
        with open(os.path.join(sys.argv[2], "CDFG.pkl"), "wb") as pkf:
            pickle.dump(cdfg, pkf)

