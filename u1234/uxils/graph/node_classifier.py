import numpy as np
import torch
from dgl import DGLGraph
from scipy import sparse
# from torch_geometric.data import Data
# from torch_geometric.utils import from_scipy_sparse_matrix

from uxils.torch_ext.graph_modules import GraphConvolutionStack
from uxils.torch_ext.module_utils import init_optimizer, init_criterion


class Graph:
    def __init__(self, pyg_data=None, dgl_graph=None):
        self.pyg_data = pyg_data
        self.dgl_graph = dgl_graph


def _masks_by_idxs(n_nodes, idxs) -> torch.tensor:
    "[1, 3, 4] -> torch.tensor([0, 1, 0, 1, 1])"
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[np.array(idxs)] = 1
    return train_mask


class ConvolutionalNodeClassifier:
    """Stack of Graph Convolution layers (GCN, TAG, SAGE, SG, Chebyshev, ...).
    Accepts layers from both dgl or pytorch_geometric.
    """

    def __init__(self, *, n_classes, conv_class, n_hiddens, in_dropout=None, out_dropout=None,
                 in_normalization=None, hidden_normalization=None, criterion='ce',
                 n_epochs=10, wd=0, lr=0.01, optimizer='adam', activation='tanh', device='cuda'):
        self.conv_class = conv_class
        self.n_hiddens = n_hiddens
        self.device = torch.device(device)
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self.lr = lr
        self.wd = wd
        self.optimizer_str = optimizer
        self.activation = activation
        self.in_normalization = in_normalization
        self.hidden_normalization = hidden_normalization
        self.model = None
        self.criterion = criterion

    def init_model(self, data):
        input_size = data.x.shape[1]
        self.model = GraphConvolutionStack(
            input_size=input_size, n_classes=self.n_classes,
            conv_class=self.conv_class, n_hiddens=self.n_hiddens,
            in_dropout=self.in_dropout, out_dropout=self.out_dropout, activation=self.activation,
            in_normalization=self.in_normalization, hidden_normalization=self.hidden_normalization)
        self.model = self.model.to(self.device)
        self.optimizer = init_optimizer(self.optimizer_str)(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.criterion = init_criterion(self.criterion)

    def fit(self, g: Graph, train_indices, n_epochs=None):
        data, g = g.pyg_data, g.dgl_graph
        train_mask = _masks_by_idxs(len(data.x), train_indices).to(self.device)

        data = data.to(self.device)
        g = g.to(self.device)
        if self.model is None:
            self.init_model(data)

        self.model.train()
        torch.set_grad_enabled(True)

        n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        for _ in range(n_epochs):
            self.optimizer.zero_grad()
            out = self.model(g, data)
            loss = self.criterion(out[train_mask], data.y[train_mask])
            loss.backward()

            self.optimizer.step()

    def predict(self, g: Graph, indices=None):
        data, g = g.pyg_data, g.dgl_graph

        g = g.to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(g, data)
            if indices is not None:
                mask = _masks_by_idxs(len(data.x), indices).to(self.device)
                pred = pred[mask]

        return pred.cpu().numpy()


def create_graph(node_features, labels, edges, edge_attr=None) -> Graph:
    assert len(node_features) == len(labels)

    node_features = torch.tensor(node_features, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    if isinstance(edges, np.ndarray):
        edge_index = torch.tensor(edges, dtype=torch.long).transpose(0, 1)
    elif isinstance(edges, sparse.spmatrix):
        edge_index, edge_weight = from_scipy_sparse_matrix(edges)
        edge_weight = edge_weight.float()
    else:
        raise ValueError('numpy(n, 2) or scipy.sparse adj')

    data = Data(node_features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
    dgl_graph = DGLGraph((data.edge_index[0], data.edge_index[1]))
    # TODO make DGLGraph lazy init
    return Graph(data, dgl_graph)
