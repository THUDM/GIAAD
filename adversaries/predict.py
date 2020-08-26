#!/user/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pickle
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import activations
from sklearn.preprocessing import StandardScaler


def predict(adj, x):

    N = adj.shape[0]
    
    # 防特殊攻击
    adj[adj>1] = 1.0
    # 数据归一化
    x = StandardScaler().fit(x).transform(x)   
    
    adj_0 = normalize_adj(adj, -0.5)
    adj_1 = normalize_adj(adj, -1.0)
    
    tf_adj_0 = sparse_adj_to_sparse_tensor(adj_0)
    tf_adj_1 = sparse_adj_to_sparse_tensor(adj_1)
    
    tf_x = tf.convert_to_tensor(x, dtype='float32')

    with open('adversaries/RGCN_weights.pkl', 'rb') as f:
        weights = pickle.load(f)
        
    label = forward([tf_x, tf_adj_0, tf_adj_1], weights).numpy().argmax(1)
    return label
   # np.savetxt(output_path, label, fmt='%i', delimiter=",")
    
def forward(inputs, weights):
    
    # 第一层
    x, *adj = inputs
    h = x @ weights[0]

    mean = activations.elu(h)
    var = activations.relu(h)

    attention = tf.exp(-var)
    mean = tf.sparse.sparse_dense_matmul(adj[0], mean * attention)
    var = tf.sparse.sparse_dense_matmul(adj[1], var * attention * attention)
    mean = activations.elu(mean)
    var = activations.elu(var)
    
    # 中间层
    i = 1
    while i<len(weights)-2:
        
        mean = activations.elu(mean @ weights[i])
        var = activations.relu(var @ weights[i+1])

        attention = tf.math.exp(-var)
        mean = tf.sparse.sparse_dense_matmul(adj[0], mean * attention)
        var = tf.sparse.sparse_dense_matmul(adj[1], var * attention * attention)

        mean = activations.elu(mean)
        var = activations.elu(var)
        i += 2
    
    # 输出层
    mean = activations.elu(mean @ weights[i])
    var = activations.relu(var @ weights[i+1])

    attention = tf.math.exp(-var)
    mean = tf.sparse.sparse_dense_matmul(adj[0], mean * attention)
    var = tf.sparse.sparse_dense_matmul(adj[1], var * attention * attention)
    
    # 采样层
    sample = tf.random.normal(tf.shape(var), 0, 1, dtype='float32')
    output = mean + tf.math.sqrt(var + 1e-8) * sample   
    
    return output
    
def normalize_adj(adjacency, rate=-0.5, add_self_loop=True):
    """Normalize adjacency matrix."""
    def normalize(adj, alpha):
        
        if add_self_loop:
            adj = adj + sp.eye(adj.shape[0])
            
        adj = adj.tocoo(copy=False)         
        row_sum = adj.sum(1).A1
        d_inv_sqrt = np.power(row_sum, alpha)
        adj.data = d_inv_sqrt[adj.row] * adj.data * d_inv_sqrt[adj.col]
        return adj.astype('float32', copy=False)

    adjacency = normalize(adjacency, rate)

    return adjacency

def sparse_adj_to_sparse_tensor(x):
    """Converts a SciPy sparse matrix to a SparseTensor."""
    sparse_coo = x.tocoo()
    row, col = sparse_coo.row, sparse_coo.col
    data, shape = sparse_coo.data, sparse_coo.shape
    indices = np.concatenate(
        (np.expand_dims(row, axis=1), np.expand_dims(col, axis=1)), axis=1)
    return tf.sparse.SparseTensor(indices, data, shape)


if __name__ == '__main__':
    args = sys.argv[-3:]
    predict(*args)
    
    

