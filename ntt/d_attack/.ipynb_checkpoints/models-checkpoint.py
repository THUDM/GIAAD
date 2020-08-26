"""
KDDCUP2020 MLTrack2
https://www.biendata.xyz/competition/kddcup_2020/

Author: NTT DOCOMO LABS
License: MIT
"""

import numpy as np

import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, GINConv
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader

from sklearn.metrics import accuracy_score

### Model Zoo ###

class GCN_simple(torch.nn.Module):
    def __init__(self, num_node_features, num_class, hidden=16, dropout_rate=0.5):
        super(GCN_simple, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden)
        self.conv2 = GCNConv(hidden, num_class)
        self.dropout_rate=dropout_rate

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__
    

class GCN(torch.nn.Module):
    """
    多層化対応モデル（AutoGraphで使われていたモデル）
    ※ Conv層を(hidden→hidden)にするため前後をLinear層で挟んでいる

    """

    def __init__(self, num_node_features=100, num_class=18, hidden=16,  dropout_rate=0.5, num_layers=2):
        super(GCN, self).__init__()
        self.first_lin = Linear(num_node_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.dropout_rate=dropout_rate

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GCN_dev(torch.nn.Module):
    """
    多層化対応モデル（AutoGraphで使われていたモデル）
    ※ Conv層を(hidden→hidden)にするため前後をLinear層で挟んでいる

    """

    def __init__(self, num_node_features=100, num_class=18, hidden=16,  dropout_rate=0.5, num_layers=2):
        super(GCN_dev, self).__init__()
        self.first_lin = Linear(num_node_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.dropout_rate=dropout_rate
        self.kWTA = kWTA(0.3)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.kWTA(self.first_lin(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        for conv in self.convs:
            x = self.kWTA(conv(x, edge_index))
            x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class SAGE(torch.nn.Module):
    """
    多層化対応モデル（AutoGraphで使われていたモデル）
    ※ Conv層を(hidden→hidden)にするため前後をLinear層で挟んでいる

    """

    def __init__(self, num_node_features=100, num_class=18, hidden=16,  dropout_rate=0.5, num_layers=2):
        super(SAGE, self).__init__()
        self.first_lin = Linear(num_node_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden, hidden, normalize=True, bias=True))
        self.lin2 = Linear(hidden, num_class)
        self.dropout_rate=dropout_rate

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Cheb(torch.nn.Module):
    """
    多層化対応モデル（AutoGraphで使われていたモデル）
    ※ Conv層を(hidden→hidden)にするため前後をLinear層で挟んでいる

    """

    def __init__(self, num_node_features=100, num_class=18, hidden=16,  dropout_rate=0.5, num_layers=2):
        super(Cheb, self).__init__()
        self.first_lin = Linear(num_node_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(ChebConv(hidden, hidden, K=2))
        self.lin2 = Linear(hidden, num_class)
        self.dropout_rate=dropout_rate

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GIN(torch.nn.Module):
    """
    多層化対応モデル（AutoGraphで使われていたモデル）
    グラフ構造が重要なので最初から畳み込み層に入力する
    """

    def __init__(self, num_node_features=100, num_class=18, hidden=16,  dropout_rate=0.5, num_layers=2, eps=0, train_eps=True):
        super(GIN, self).__init__()
        self.first_conv = GINConv(Sequential(Linear(num_node_features, hidden), ReLU(), Linear(hidden, hidden)), eps, train_eps)
        self.first_bn = BatchNorm1d(hidden)
        self.nns = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers):
            self.nns.append(Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden)))
            self.bns.append(BatchNorm1d(hidden))
            self.convs.append(GINConv(self.nns[i], eps, train_eps))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_class)
        self.dropout_rate=dropout_rate

    def reset_parameters(self):
        self.first_conv.reset_parameters()
        self.first_bn.reset_parameters()
        for nn, conv, bn in zip(self.nns, self.convs, self.bns):
            nn.reset_parameters()
            conv.reset_parameters()
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        #x = F.relu(self.first_lin(x))
        x = F.relu(self.first_conv(x, edge_index))
        x = self.first_bn(x)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(conv(x, edge_index))
            x = bn(x)
            #x = F.dropout(x, self.dropout_rate, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


####



def predict(model, data, device):
    """推論したラベルの確率を算出する関数"""
    model.eval()
    with torch.no_grad():
        out = model(data.to(device))
        _, pred = out.max(dim=1)
    return out, pred


def test(model, data, device):
    """Acc, Lossを算出する関数"""
    model.eval()
    out, pred = predict(model, data, device)
    loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask].to(device))
    acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
    return acc, loss


def model_train(data, params, device, kind):
    """earlystopping"""
    # GCNの設定
    print("Strat model_train...")
    print(f"model name: {kind}")
    print(f"parameter: {params}")
    model = eval(kind)(data.num_node_features,
                       data.num_class,
                       params["hidden"],
                       params["dropout_rate"],
                      ).to(device)

    # Optimizerを設定
    optimizer =torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=5e-4)

    # EarlyStoppingを初期化
    early_stopping = EarlyStopping(patience=params["patience"], verbose=False)
    
    # 学習
    for epoch in np.arange(4000):
        train_loss = train(model, data, device, optimizer)
        val_acc, val_loss = test(model, data, device)
        if epoch%10==0:
            print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('./tmp/checkpoint.pt', map_location=device))

    return model


def train(model, data, device, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.to(device))
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss


def model_batch_train(dataset, params, device, kind):
    """earlystopping"""
    # GCNの設定
    model = eval(kind)(dataset.pyg_data.num_node_features,
                dataset.pyg_data.num_class,
                params["hidden"],
                params["dropout_rate"]).to(device)

    # Optimizerを設定
    optimizer =torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=5e-4)

    # EarlyStoppingを初期化
    early_stopping = EarlyStopping(patience=params["patience"], verbose=False)
    
    train_loader = DataLoader(dataset, batch_size=1)
    # 学習
    for epoch in np.arange(4000):
        train_loss = batch_train(model, train_loader, device, optimizer)
        val_acc, val_loss = test(model, dataset.pyg_data, device)
        if epoch%10==0:
            print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('./tmp/checkpoint.pt'))

    return model



def batch_train(model, loader, device, optimizer):
    """DataLoaderクラスからミニバッチを取り出してtrainする"""
    model.train()    
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = F.nll_loss(out, data.y.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='./tmp/checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if(self.counter%10==0):
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class kWTA(torch.nn.Module):
    def __init__(self, sr=0.2):
        super(kWTA, self).__init__()
        self.sr = sr
    # Paper's forward implementation
    def forward(self, x):
        tmpx = x.view(x.shape[0], -1)
        size = tmpx.shape[1]
        k = int(self.sr * size)
        topval = tmpx.topk(k, dim=1)[0][:,-1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1,0).view_as(x)
        comp = (x>=topval).to(x)
        return comp*x