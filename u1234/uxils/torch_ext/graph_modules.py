import torch.nn as nn
from dgl import DGLGraph

from .module_utils import init_activation, init_normalization

WITH_EDGE_WEIGHTS = ['ChebConv', 'GCNConv', 'SAGEConv', 'GraphConv', 'TAGConv', 'ARMAConv']


def _get_name(obj):
    if hasattr(obj, 'func'):
        return str(obj.func)
    else:
        return str(obj)


def _is_dgl(obj):
    # TODO check dgl, pyg or raise
    if hasattr(obj, 'func'):
        obj = obj.func
    module_name = str(obj.__module__)
    return 'dgl' in module_name


def with_edge_weights(conv_mod):
    for c in WITH_EDGE_WEIGHTS:
        if f'.{c}' in str(type(conv_mod)):
            return True
    return False


class GraphConvolutionStack(nn.Module):
    """
            x
            |
            FC
            |
    Sequence of GraphConv
            |
            FC
            |
            out
    """
    def __init__(
            self, input_size, n_classes, conv_class,
            in_dropout, out_dropout, n_hiddens, activation, in_normalization, hidden_normalization):
        super().__init__()

        self.layers = nn.ModuleList()
        self.in_nn = nn.Linear(input_size, n_hiddens[0])
        n_prev = n_hiddens[0]
        self.hidden_normalization = nn.ModuleList() if hidden_normalization else None

        for n_hidden in n_hiddens[1:]:
            if 'GINConv' in _get_name(conv_class):
                self.layers.append(conv_class(
                    apply_func=nn.Linear(n_prev, n_hidden),
                    aggregator_type='mean'))
            elif 'AGNNConv' in _get_name(conv_class):
                self.layers.append(conv_class())
            elif 'APPNPConv' in _get_name(conv_class):
                self.layers.append(nn.Linear(n_prev, n_hidden))
            else:
                self.layers.append(conv_class(n_prev, n_hidden))

            n_prev = n_hidden
            if hidden_normalization is not None:
                self.hidden_normalization.append(init_normalization(hidden_normalization)(n_hidden))

        self.out_nn = nn.Linear(n_hiddens[-1], n_classes)
        self.in_dropout = nn.Dropout(in_dropout) if in_dropout is not None else None
        self.out_dropout = nn.Dropout(out_dropout) if out_dropout is not None else None
        self.activation = init_activation(activation)
        self.in_normalization = init_normalization(in_normalization)(input_size) if in_normalization else None

    def forward(self, g: DGLGraph, data):
        x = data.x

        if self.in_normalization is not None:
            x = self.in_normalization(x)

        x = self.in_nn(x)
        x = self.activation(x)
        if self.in_dropout is not None:
            x = self.in_dropout(x)

        for idx, layer in enumerate(self.layers):
            if _is_dgl(layer):
                x = layer(g, x)
            else:
                if with_edge_weights(layer):
                    x = layer(x, edge_index=data.edge_index, edge_weight=data.edge_weight)
                else:
                    x = layer(x, edge_index=data.edge_index)

            if self.hidden_normalization is not None:
                x = self.hidden_normalization[idx](x)

            x = self.activation(x).squeeze()  # GAT num heads (b, H, d)

        if self.out_dropout is not None:
            x = self.out_dropout(x)

        x = self.out_nn(x)
        return x
