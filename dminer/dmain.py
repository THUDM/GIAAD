import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import dgl
from dgl import DGLGraph
from dgl.nn.pytorch.conv import SAGEConv
from dgl.transform import add_self_loop
#import scipy.sparse as sps
#from scipy.sparse import csr_matrix
#import random


import sys
import time

class GraphSAGE(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = torch.nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h
    def embed(self,g,features):
        h = features
        
        for i in range(len(self.layers) -1):
            layer = self.layers[i]
            h = layer(g, h)
        return h 
class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits    
def evaluate_labels(labels):
    true_labels = np.load('train_labels_formal_stage.npy') #- 2
    print(true_labels)
    #true_labels[ np.where(true_labels<0)] = 0
    mask = np.arange(0,609574)
    labels = labels[mask]
    correct = np.sum(true_labels == labels)
    print('Acc', correct * 1.0 / len(labels))    
'''
def predict(adj,features):
    #adj_mat = pickle.load(open('adj_matrix_formal_stage.pkl', 'rb'))
    #feats = np.load('feature_formal_stage.npy')
    #true_labels = np.load('train_labels_formal_stage.npy') - 2
    #true_labels[ np.where(true_labels<0)] = 0
    
    t1 = time.time()
    #adj_mat = pickle.load(open(sys.argv[1],'rb'))
    #feats = np.load(sys.argv[2])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = torch.FloatTensor(features).to(dev)
    features = features / (features.norm(dim=1)[:, None] + 1e-8)
    
    graph = DGLGraph(adj)
    graph = add_self_loop(graph)
    
    
    model_filename = 'dminer/def_pool2.pt'

    model = torch.load(model_filename,map_location=dev)
    
    model.eval()
    
    with torch.no_grad():
        logits = model(graph, features)
        #logits = logits[mask]
        _, labels = torch.max(logits, dim=1)
        
    labels = labels.cpu().detach().numpy()
    labels[ np.where(labels==0)] = -1
    labels = labels + 2
    
    return labels
    #print(labels)
    
    
'''
