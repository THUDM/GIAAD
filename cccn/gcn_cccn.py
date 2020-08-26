import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import scipy.sparse as sp
import numpy as np
import torch.optim as optim
from torch.nn.modules.module import Module
from copy import deepcopy
import time

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0, lr=0.01, weight_decay=5e-5, with_relu=True, with_bias=True, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device # cpu或gpu
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nhid, with_bias=with_bias)
        self.gc3 = GraphConvolution(nhid, nclass, with_bias=with_bias)

        
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def forward(self, x, adj):
        '''
            adj: normalized adjacency matrix
        '''
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.gc2(x, adj))

        else:
            x = self.gc1(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

    def test(self, idx_test):
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test

    def predict(self, features=None, adj=None):
        '''By default, inputs are normalized data'''

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            self.features = features
            return self.forward(self.features, adj)
        

class GCN_norm(nn.Module):
    def __init__(self,nhid,nfeat,nclass,device):
        super(GCN_norm,self).__init__()
        self.GCN=GCN(nfeat, nhid, nclass,device=device)
    def forward(self,x,adj,dropout=0):
        x=F.relu(x+1.8)-1.8
        x=1.8-F.relu(1.8-x)
        x=self.GCN(x,adj,dropout)
        return x
       
class GCN_norm2(nn.Module):
    def __init__(self,load_GCN):
        super(GCN_norm2,self).__init__()
        self.GCN=load_GCN
    def forward(self,x,adj):
        x=F.relu(x+1.8)-1.8
        x=1.8-F.relu(1.8-x)
        x=self.GCN(x,adj)
        return x
    
def GCNadj(adj, features):
    adj.data[adj.data != 1] = 1
    
#     adj=sp.eye(adj.shape[0])+adj
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     adj=adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    adj = adj + adj.T # 对称化
    adj = adj.tolil() # 转化提高效率
    adj[adj > 1] = 1 # 权值设置为1
    # whether to set diag=0?
    adj.setdiag(0) # 无自环
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros() # 稀疏矩阵去0元素
    A_hat = adj
    degree = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = sp.diags(degree)

    I = sp.diags(np.ones(adj.shape[0]))

    
    degree_reverse = degree ** (-1 / 2)
    degree_reverse[np.isinf(degree_reverse)] = 0.
    D_hat_reverse = sp.diags(degree_reverse)
    # adj = D_hat +A_hat
#     adj = D_hat_reverse * (D_hat + A_hat) * D_hat_reverse # 与官方的加自环相比，这里是直接加该节点的度
    adj = D_hat_reverse * (I + A_hat) * D_hat_reverse 

    
    adj = adj.tocoo()
    return adj



def get_cuda():
    use_cuda = True
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            use_cuda = False
    else:
        device = torch.device('cpu')
        use_cuda = False
    return device,use_cuda