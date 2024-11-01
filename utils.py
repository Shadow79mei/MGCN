"""Data reading tools."""

import json
import torch
import numpy as np
import networkx as nx
from scipy import sparse
from texttable import Texttable
import math
from scipy.interpolate import RegularGridInterpolator

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def create_propagator_matrix(A, args):

    A = sparse.coo_matrix(A, dtype=np.float32)
    I = sparse.eye(A.shape[0])
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)  #.dot函数的作用是获取两个元素a,b的乘积
    propagator = dict()
    A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    #print(ind.shape)  # Bay:(3958, 2)  (4058, 2)
    propagator["indices"] = torch.LongTensor(ind.T).to(args.device)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data).to(args.device)
    print('propagator["values"].shape', propagator["values"].shape)  # Bay:torch.Size([3958])   torch.Size([4058])

    return propagator

def feature_reader(features, args):
    """
    :return out_features: Dict with index and value tensor.
    """
    features = sparse.coo_matrix(features, dtype=np.float32)
    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    #reshape(-1,1)转换成1列
    #print(ind.shape)  #bay:(835072, 2)  (835072, 2)
    out_features["indices"] = torch.LongTensor(ind.T).to(args.device)
    out_features["values"] = torch.FloatTensor(features.data).to(args.device)
    out_features["dimensions"] = features.shape
    print('out_features["values"].shape', out_features["values"].shape)

    return out_features

def target_reader(data_gt, Seg, superpixel_count, args):

    segments = np.reshape(Seg, [-1])
    data_gt = np.reshape(data_gt, [-1])
    new_gt = np.zeros(superpixel_count, dtype=np.float32)
    for i in range(superpixel_count):
        idx = np.where(segments == i)[0]
        count = len(idx)
        new_gt[i] = sum(data_gt[idx]) / count
        if new_gt[i] > 1:
            new_gt[i] = 1
        if new_gt[i] < 0:
            new_gt[i] = 0

    target = torch.LongTensor(new_gt)

    return target

def createA(A, args):

    I = sparse.eye(A.shape[0])
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)  #.dot函数的作用是获取两个元素a,b的乘积

    propagator = torch.from_numpy(A_tilde_hat).to(args.device)

    return propagator

def createH(features, args):

    out_features = torch.from_numpy(features).to(args.device)

    return out_features
