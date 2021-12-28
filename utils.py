import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
import collections
from collections import Counter
import numpy as np
from multiprocessing import Process, Queue
import pandas as pd
import os
import random


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

def translation(data, item_dic):

    datax = []
    for i in range(len(data[0])):
        datax.append([item_dic[s] for s in data[0][i]])
    datay = [item_dic[s] for s in data[1]]

    return (datax, datay)

class Data():
    def __init__(self, data, window):
        inputs = data[0]
        self.inputs = np.asarray(inputs) 
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.window = window


    def generate_batch(self, batch_size, shuffle = False):
        if shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, iList):
        inputs, targets = self.inputs[iList], self.targets[iList]
        items, n_node, H, HT, G, EG, alias_inputs, node_masks, node_dic = [], [], [], [], [], [], [], [], []
        num_edge, edge_mask, edge_inputs = [], [], []

        for u_input in inputs:
            temp_s = u_input
            
            temp_l = list(set(temp_s))    
            temp_dic = {temp_l[i]: i for i in range(len(temp_l))}        
            n_node.append(temp_l)
            alias_inputs.append([temp_dic[i] for i in temp_s])
            node_dic.append(temp_dic)

            min_s = min(self.window, len(u_input))
            num_edge.append(int((1 + min_s) * len(u_input) - (1 + min_s) * min_s / 2))


        max_n_node = np.max([len(i) for i in n_node])

        max_n_edge = max(num_edge)

        max_se_len = max([len(i) for i in alias_inputs])

        edge_mask = [[1] * len(le) + [0] * (max_n_edge - len(le)) for le in alias_inputs]

        for idx in range(len(inputs)):
            u_input = inputs[idx]
            effect_len = len(alias_inputs[idx])
            node = n_node[idx]
            items.append(node + (max_n_node - len(node)) * [0])

            effect_list = alias_inputs[idx]
            ws = np.ones(max_n_edge)
            cols = []
            rows = []
            edg = []
            e_idx = 0

            for w in range(1 + min(self.window, effect_len-1)):
                edge_idx = list(np.arange(e_idx, e_idx + effect_len-w))
                edg += edge_idx
                for ww in range(w + 1):
                    rows += effect_list[ww:ww+effect_len-w]
                    cols += edge_idx

                e_idx += len(edge_idx)


            u_H = sp.coo_matrix(([1.0]*len(rows), (rows, cols)), shape=(max_n_node, max_n_edge))
            HT.append(np.asarray(u_H.T.todense()))


            node_masks.append((max_se_len - len(alias_inputs[idx])) * [0] + [1]*len(alias_inputs[idx]))
            alias_inputs[idx] = (max_se_len - len(alias_inputs[idx])) * [0] + alias_inputs[idx]


            edge_inputs.append(edg + (max_n_edge - len(edg))*[0])

        return alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs

