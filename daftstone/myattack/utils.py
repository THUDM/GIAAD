"""
Implementation of the method proposed in the paper:
'Adversarial Attacks on Graph Neural Networks via Meta Learning'
by Daniel Z端gner, Stephan G端nnemann
Published at ICLR 2019 in New Orleans, USA.
Copyright (C) 2019
Daniel Z端gner
Technical University of Munich
"""
import pickle
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def generate_node(length, begin=609574):
    end = begin + 50000
    addnode = np.zeros((500, length))
    count = [[] for i in range(18)]
    label = np.load("label.npy")[begin:end]
    for i in range(50000):
        count[label[i]].append(i)
    idx = []
    sum=np.zeros(18)
    for i in range(18):
        sum[i]=len(count[i])
    idx1=np.argsort(sum)
    for i in idx1:
        temp=count[i]
        np.random.shuffle(temp)
        idx += temp
    idx = np.array(idx) + begin
    # idx = np.arange(begin, end)
    # np.random.shuffle(idx)
    for i in range(500):
        for j in range(100):
            addnode[i, idx[i * 100 + j]] = 1
    print("test_random", np.min(np.sum(addnode[:, begin:end], axis=-1)))
    print(np.min(np.sum(addnode[:, begin:end], axis=0)))
    return sp.csr_matrix(addnode), np.arange(begin, end)


def sparse_to_tuple1(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def load_pkl():
    with open("../temp/data/adj_matrix_formal_stage.pkl", "rb+") as f:
        adj = pickle.load(f)

    feat=np.load("../temp/data/feature_formal_stage.npy")
    label=np.load("../temp/data/train_labels_formal_stage.npy")
    return adj, feat, label


def sparse_to_tuple(norm_adj, idx):
    begin=609574
    rows1 = []
    cols1 = []
    for i in idx:
        for j in range(begin, begin+50000):
            rows1.append(j)
            cols1.append(i)

    rows1 = np.array(rows1, np.int)
    cols1 = np.array(cols1, np.int)
    data1 = np.reshape(norm_adj[idx, :][:, begin:begin+50000].toarray(), -1)

    t2 = norm_adj.copy()
    t2[begin:begin+50000, :][:, idx] = 0
    t2.eliminate_zeros()
    t2 = t2.tocoo()
    rows2 = t2.row
    cols2 = t2.col
    data2 = t2.data
    # print(rows1.shape,cols1.shape,data1.shape,rows2.shape,cols2.shape,data2.shape)
    coords = np.vstack((np.concatenate([rows1, rows2], 0), np.concatenate([cols1, cols2], 0))).transpose()
    shape = norm_adj.shape
    return coords, data1, data2, shape

def preprocess_graph(adj):
    """
    Perform the processing of the adjacency matrix proposed by Kipf et al. 2017.

    Parameters
    ----------
    adj: sp.spmatrix
        Input adjacency matrix.

    Returns
    -------
    The matrix (D+1)^(-0.5) (adj + I) (D+1)^(-0.5)

    """
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized

