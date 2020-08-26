import os
import sys
import pickle
import time

import dgl
import numpy as np
import scipy.sparse as sp
import torch as th
import torch.nn.functional as F

import torch.nn as nn
from dgl.nn.pytorch.conv import TAGConv


class TAGCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(TAGCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(TAGConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(TAGConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(TAGConv(n_hidden, n_classes))  # activation=None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        return h


def adj_preprocess(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(axis=1).A1
    deg = sp.diags(rowsum ** (-0.5))
    adj_ = deg @ adj_ @ deg.tocsr()

    return adj_



def predict(adj,features):
    
   # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dev = th.device('cpu')
    '''
    fg = open(sys.argv[1], 'rb')
    adj = pickle.load(fg)
    features = np.load(sys.argv[2])
    '''
    #graph = dgl.DGLGraph()
    adj = adj_preprocess(adj)
    #graph.from_scipy_sparse_matrix(adj)
    graph=dgl.from_scipy(adj)
    features = th.FloatTensor(features).to(dev)
    features[th.where(features < -1.0)[0]] = 0
    features[th.where(features > 1.0)[0]] = 0
    features = 2 * th.atan(features) / th.Tensor([np.pi]).to(dev)
    graph.ndata['features'] = features

    model = TAGCN(100, 128, 20, 3, activation=F.leaky_relu, dropout=0.0)
    model_states = th.load('speit/model.pkl', map_location=dev)
    model.load_state_dict(model_states)
    model = model.to(dev)
    model.eval()

    logits = model(graph, features)
    pred = logits.argmax(1)

    return pred.cpu().numpy()
    


