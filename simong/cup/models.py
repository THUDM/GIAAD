from collections import OrderedDict
from itertools import chain
import math
from typing import List, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, APPNP, MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

def normg(edge_index, num_nodes, edge_weight=None, improved=False,
     dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class ChainableGCNConv(GCNConv):
    """Simple extension to allow the use of `nn.Sequential` with `GCNConv`. The arguments are wrapped as a Tuple/List
    are are expanded for Pytorch Geometric.

    Parameters
    ----------
    See https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.conv.gcn
    """

    def forward(self, arguments: Sequence[torch.Tensor] = None) -> torch.Tensor:
        """Predictions based on the input.

        Parameters
        ----------
        arguments : Sequence[torch.Tensor]
            [x, edge indices] or [x, edge indices, edge weights], by default None

        Returns
        -------
        torch.Tensor
            the output of `GCNConv`.

        Raises
        ------
        NotImplementedError
            if the arguments are not of length 2 or 3
        """
        if len(arguments) == 2:
            x, edge_index = arguments
            edge_weight = None
        elif len(arguments) == 3:
            x, edge_index, edge_weight = arguments
        else:
            raise NotImplementedError("This method is just implemented for two or three arguments")
        return super(ChainableGCNConv, self).forward(x, edge_index, edge_weight=edge_weight)


class GCN(nn.Module):
    """Two layer GCN implementation.

    Parameters
    ----------
    n_features : int
        Number of attributes for each node
    n_classes : int
        Number of classes for prediction
    hidden_dimensions: List[int]
        Internal number of features. `len(hidden_dimensions)` defines the number of hidden representations.
    activation : nn.Module, optional
        Arbitrary activation function for the hidden layer, by default nn.ReLU()
    dropout : int, optional
        Dropout rate, by default 0.5
    """

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 hidden_dimensions: List[int] = [64],
                 activation: nn.Module = nn.ReLU(),
                 dropout: int = 0.5):
        super().__init__()
        self.n_features = n_features
        self.hidden_dimensions = hidden_dimensions
        self.n_classes = n_classes
        self.state = {}
        self._activation = activation
        self._dropout = dropout
        self.layers = self._build_layers()

    def _build_layers(self):
        return nn.ModuleList([
            *[nn.Sequential(OrderedDict([
                (f'gcn_{idx}', ChainableGCNConv(in_channels=in_channels, out_channels=out_channels)),
                (f'activation_{idx}', self._activation),
                (f'dropout_{idx}', nn.Dropout(p=self._dropout))
            ]))
                for idx, (in_channels, out_channels)
                in enumerate(zip([self.n_features] + self.hidden_dimensions[:-1], self.hidden_dimensions))
            ],
            nn.Sequential(OrderedDict([
                (f'gcn_{len(self.hidden_dimensions)}', ChainableGCNConv(
                    in_channels=self.hidden_dimensions[-1], out_channels=self.n_classes))

            ]))
        ])

    def forward(self,
                features: torch.Tensor,
                edge_idx: torch.Tensor,
                edge_weight: torch.Tensor = None,
                n: int = None,
                d: int = None) -> torch.Tensor:

        x = features
        for layer in self.layers:
            x = layer((x, edge_idx, edge_weight))

        return x

    def _normalize(self, A: torch.sparse.FloatTensor) -> torch.tensor:
        """
        For calculating $\hat{A} = 𝐷^{−\frac{1}{2}} 𝐴 𝐷^{−\frac{1}{2}}$.

        Parameters
        ----------
        A: torch.sparse.FloatTensor
            Sparse adjacency matrix with added self-loops.

        Returns
        -------
        A_hat: torch.sparse.FloatTensor
            Normalized message passing matrix
        """
        row, col = A._indices()
        edge_weight = A._values()
        deg = (A @ torch.ones(A.shape[0], 1, device=A.device)).squeeze()
        deg_inv_sqrt = deg.pow(-0.5)
        normalized_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        A_hat = torch.sparse.FloatTensor(A._indices(), normalized_edge_weight, A.shape)
        return A_hat


class APPNPDropout(APPNP):

    def __init__(self, dropout: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index,
                edge_weight=None) -> torch.Tensor:

        edge_index, norm =normg(
        edge_index, x.size(0), edge_weight, dtype=x.dtype)

        hidden = x
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=F.dropout(norm, self.dropout, self.training))
            x = x * (1 - self.alpha)
            x = x + self.alpha * hidden

        return x


class APPNPModel(torch.nn.Module):
    """APPNP implementation.

    Parameters
    ----------
    n_features : int
        Number of attributes for each node
    n_classes : int
        Number of classes for prediction
    hidden_dimensions: List[int]
        Internal number of features. `len(hidden_dimensions)` defines the number of hidden representations.
    activation : nn.Module, optional
        Arbitrary activation function for the hidden layer, by default nn.ReLU()
    dropout : int, optional
        Dropout rate, by default 0.5
    """

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 hidden_dimensions: List[int] = [64],
                 activation: nn.Module = nn.ReLU(),
                 dropout: int = 0.5,
                 do_use_dropout_for_propagation: bool = False,
                 alpha: float = 0.1,
                 n_propagation: int = 10):
        super().__init__()
        if len(hidden_dimensions) > 0:
            self._transform_features = nn.Sequential(OrderedDict([
                (f'dropout_{0}', nn.Dropout(p=dropout))
            ] + list(chain(*[
                [(f'linear_{idx}', nn.Linear(in_features=in_features, out_features=out_features)),
                 (f'activation_{idx}', activation)]
                for idx, (in_features, out_features)
                in enumerate(zip([n_features] + hidden_dimensions[:-1], hidden_dimensions))
            ])) + [
                (f'linear_{len(hidden_dimensions)}', nn.Linear(in_features=hidden_dimensions[-1],
                                                               out_features=n_classes)),
                (f'dropout_{len(hidden_dimensions)}', nn.Dropout(p=dropout)),
            ]))
        else:
            self._transform_features = nn.Sequential(OrderedDict([
                (f'dropout_{0}', nn.Dropout(p=dropout)),
                (f'linear_{len(hidden_dimensions)}', nn.Linear(in_features=n_features,
                                                               out_features=n_classes)),
                # (f'dropout_{len(hidden_dimensions)}', nn.Dropout(p=dropout)),
            ]))
        self._propagate = APPNPDropout(K=n_propagation, alpha=alpha,
                                       dropout=dropout if do_use_dropout_for_propagation else 0)

    def forward(self,
                features: torch.Tensor,
                edge_idx: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        logits = self._transform_features(features)
        logits = self._propagate(logits, edge_idx,edge_weight=edge_weight)
        return logits


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, edge_idx: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        if edge_weight is None:
            edge_weight = torch.ones_like(edge_idx[0]).float()
        edge_idx, edge_weight = normg(edge_idx, x.size(0), edge_weight, dtype=x.dtype)
        adj = torch.sparse.FloatTensor(edge_idx, edge_weight, (x.shape[0], x.shape[0]))
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner  # F.log_softmax(layer_inner, dim=1)


class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant, residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, edge_idx: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        if edge_weight is None:
            edge_weight = torch.ones_like(edge_idx[0]).float()
        edge_idx, edge_weight = normg(edge_idx, x.size(0), edge_weight, dtype=x.dtype)
        adj = torch.sparse.FloatTensor(edge_idx, edge_weight, (x.shape[0], x.shape[0]))
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner  # layer_inner
