from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append("daftstone/")
from myattack.robust_GCN import Robust_GCN
import myattack.utils as utils

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import pickle
import time

start = time.time()

def validate(adj,features):

    length = 659574

    _A_obs = adj
    _X_obs = features
    _A_obs = _A_obs.astype("float32")
    _A_obs.eliminate_zeros()
    _X_obs = _X_obs.astype("float32")
    origin_length = _X_obs.shape[0]
    dif = 660074 - origin_length
    if (origin_length < 660074):
      #  print("test")
        add1 = sp.csr_matrix(np.zeros((dif, origin_length)))
        _A_obs = sp.vstack([_A_obs, add1])
        A_temp = sp.csr_matrix(np.zeros((660074, dif)))
        _A_obs = sp.hstack([_A_obs, A_temp])
        _X_obs = np.concatenate([_X_obs, np.ones((dif, 100)) * 1.8])

    _Z_obs_true = np.zeros((1, 18))

    dropout = 0.2
    learning_rate = 0.005
    split_val = np.arange(500000, length - 50000)
    split_test = np.arange(length - 50000, 659574)

    alpha, belta = 35, 60
    p = [0.5, 0.3]

    # calculate negative attributes
    _X_obs = -_X_obs
    _X_obs[np.isnan(_X_obs)] = 2.0
    _X_obs[np.isinf(_X_obs)] = 2.0

    # do detection1
    x = np.abs(_X_obs)
    a = np.sum(x > p[0], axis=1)
    b = np.sum(x > p[1], axis=1)
    idx = (a > alpha) + (b > belta)

    # do detection2 // select top-k item
    m1 = np.max(x, axis=1)
    idx1 = np.argmax(x, axis=1)
    for i in range(idx1.shape[0]):
        x[i, idx1[i]] = 0
    idx1 = np.argmax(x, axis=1)
    for i in range(idx1.shape[0]):
        x[i, idx1[i]] = 0
    m2 = np.max(x, axis=1)
    idx1 = m1 - m2 <= 0.002
    idx2 = np.where(m1 == 0)[0]
    idx1[idx2] = False

    # do detection3 //use dispersion of data
    scale = 1.5
    dispersion = np.load("daftstone/max_dispersion.npy")
    idx3 = np.sum(np.abs(_X_obs) > dispersion * scale, axis=1) !=0

    # combine
    idx = np.where(idx + idx1 + idx3)[0]

  #  print(time.time() - start)
  #  print(idx.shape)
    if (len(idx) != 0):
        _X_obs[idx,] = 0
    flag = np.zeros((660074), dtype=np.int)
    if (len(idx) != 0):
        flag[idx] = 1

    _A_obs = _A_obs.tocoo()
    row = _A_obs.row
    col = _A_obs.col
    data = _A_obs.data
    data[flag[row] == 1] = 0
    data[flag[col] == 1] = 0
    _A_obs = sp.coo_matrix((data, (row, col)), shape=(660074, 660074))

    _A_obs_norm = utils.preprocess_graph(_A_obs)
    _A_obs_tuple = utils.sparse_to_tuple1(_A_obs_norm)
    myattack = Robust_GCN(_A_obs_tuple, _X_obs, _Z_obs_true, gpu_id=None, learning_rate=learning_rate,
                          dropout=dropout)
    preds = myattack.get_logits(_A_obs_tuple, _X_obs)
    pred = preds.argmax(1)
    
    pred[pred == 0] = 18
    pred[pred == 2] = 19

    return pred[0:origin_length]
