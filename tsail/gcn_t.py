import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import scipy.sparse as sp
import numpy as np

def sparse_dense_mul(a,b,c):
    i=a._indices()[1]
    j=a._indices()[0]
    v=a._values()
    newv=(b[i]+c[j]).squeeze()
    newv=torch.exp(F.leaky_relu(newv))
    
    new=torch.sparse.FloatTensor(a._indices(), newv, a.size())
    return new
    
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,activation=None,dropout=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll=nn.Linear(in_features,out_features).cuda()
        
        self.activation=activation
        self.dropout=dropout
    def forward(self,x,adj,dropout=0):
        
        x=self.ll(x)
        x=torch.spmm(adj,x)
        if not(self.activation is None):
            x=self.activation(x)
        if self.dropout:
            x=F.dropout(x,dropout)
        return x
class MLPLayer(nn.Module):

    def __init__(self, in_features, out_features,activation=None,dropout=False):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll=nn.Linear(in_features,out_features).cuda()
        
        self.activation=activation
        self.dropout=dropout
    def forward(self,x,adj,dropout=0):
        
        x=self.ll(x)
      #  x=torch.spmm(adj,x)
        if not(self.activation is None):
            x=self.activation(x)
        if self.dropout:
            x=F.dropout(x,dropout)
        return x
class GAThead(nn.Module):
    def __init__(self, in_features, dims,activation=F.leaky_relu):
        super(GAThead, self).__init__()
        self.in_features = in_features
        self.dims = dims
        self.ll=nn.Linear(in_features,dims).cuda()
        # here we get of every number
        self.ll_att=nn.Linear(dims,1).cuda()
        #self.ll_att2=nn.Linear(dims,1).cuda()
        self.special_spmm = SpecialSpmm()
        self.activation=activation
       
    def forward(self,x,adj,dropout=0):
        x=F.dropout(x,dropout)
        x=self.ll(x)
        value=self.ll_att(x)
        #value2=self.ll_att2(x)
        value=F.leaky_relu(value)
        value=20-F.leaky_relu(20-value)
        #print(value.max())
        value=torch.exp(value)
        #print(value.max())
       # value=sparse_dense_mul(adj,value,value2)
        
        #dividefactor=torch.sparse.sum(value,dim=1).to_dense().unsqueeze(1)
        
        dividefactor=torch.spmm(adj,value)
        #print(dividefactor.max(),dividefactor.min())
        x=x*value
        x=torch.spmm(adj,x)
        #print(x.shape,dividefactor.shape)
        #print((x!=x).sum())
        #print((dividefactor!=dividefactor).sum())
        x=x/dividefactor
        #print((x!=x).sum())
        if self.activation!=None:
            x=self.activation(x)
        return x

class GATLayer(nn.Module):
    def __init__(self, in_features, n_heads,dims,activation=None,type=0):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.n_heads=n_heads
        self.dims = dims
        self.heads=nn.ModuleList()
        self.type=type
        for i in range(n_heads):
            self.heads.append(GAThead(in_features,dims,activation=activation))
            
        
    def forward(self,x,adj,dropout=0):
        xp=[]
        for i in range(self.n_heads):
            xp.append(self.heads[i](x,adj,dropout))
        #n*8
        if self.type==0:
            sum=torch.cat(xp,1)
        else:
            sum=torch.sum(torch.stack(xp),0)/self.n_heads
        
        return sum
class GAT(nn.Module):
    def __init__(self,num_layers,num_heads,head_dim):
        super(GAT, self).__init__()
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.head_dim=head_dim
        
        self.layers=nn.ModuleList()
        #print(num_layers)
        
        
        for i in range(num_layers):
            if i!=num_layers-1:
                self.layers.append(GATLayer(num_heads[i]*head_dim[i],num_heads[i+1],head_dim[i+1],activation=F.elu))
            else:
                self.layers.append(GATLayer(num_heads[i]*head_dim[i],num_heads[i+1],head_dim[i+1],type=1))
    def forward(self,x,adj,dropout=0):
        for layer in self.layers:
             x=layer(x,adj,dropout=dropout)
        # x=F.softmax(x, dim=-1)
        return x

class MLP(nn.Module):
    def __init__(self,num_layers,num_features):
        super(MLP, self).__init__()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
        #print(num_layers)
        
        for i in range(num_layers):
            if i!=num_layers-1:
                self.layers.append(MLPLayer(num_features[i],num_features[i+1],activation=F.elu,dropout=True).cuda())
            else:
                self.layers.append(MLPLayer(num_features[i],num_features[i+1]).cuda())
        #print(self.layers)
        
    def forward(self,x,adj,dropout=0):
        for layer in self.layers:
            x=layer(x,adj,dropout=dropout)
       # x=F.softmax(x, dim=-1)
        return x
class GCN(nn.Module):
    def __init__(self,num_layers,num_features):
        super(GCN, self).__init__()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
        #print(num_layers)
        
        for i in range(num_layers):
            if i!=num_layers-1:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1],activation=F.elu,dropout=True).cuda())
            else:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1]).cuda())
        #print(self.layers)
        
    def forward(self,x,adj,dropout=0):
        for layer in self.layers:
            x=layer(x,adj,dropout=dropout)
       # x=F.softmax(x, dim=-1)
        return x

class GCN_norm(nn.Module):
    def __init__(self,num_layers,num_features):
        super(GCN_norm,self).__init__()
        self.GCN=GCN(num_layers,num_features)
    def forward(self,x,adj,dropout=0):
        x=F.relu(x+1.8)-1.8
        x=1.8-F.relu(1.8-x)
        x=self.GCN(x,adj,dropout)
        return x
        
def GCNadj(adj):
    #adj=sp.eye(adj.shape[0])+adj
    adj = sp.coo_matrix(adj)
    # rowsum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # adj=adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return adj
