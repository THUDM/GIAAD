import sys

sys.path.append('../..')
import random
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GATConv, GMMConv, LEConv, GINConv, SAGEConv  # noqa
import torch
import torch.nn as nn
import numpy as np

N_origin_nodes = 659574
N_train_nodes = 609574
N_fake_nodes = 500
N_feats = 100
N_labels = 19
N_fake_edges = 100


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(N_feats,128))
        for i in range(15):
            self.layers.append(nn.Sequential(
                nn.Linear(128,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            ))
        self.layers.append(nn.Linear(128,N_labels))


    def forward(self, x, edge_idex):
        x = self.layers[0](x)
        for i in range(1,len(self.layers)-1):
            x = self.layers[i](x) + x

        x = self.layers[len(self.layers)-1](x)

        return x


# GCN

class Pyg_Net(torch.nn.Module):
    def __init__(self):
        super(Pyg_Net, self).__init__()
        self.conv1 = GCNConv(N_feats, N_feats, cached=True,
                             normalize=True)
        self.conv2 = GCNConv(N_feats, N_feats, cached=True,
                             normalize=True)
        self.conv3 = GCNConv(N_feats, 256, cached=True,
                             normalize=True)
        # self.conv4 = GCNConv(N_feats, N_feats, cached=True,
        #                      normalize=True)

        self.fc1 = nn.Linear(256, 256)
        # self.fc2 = nn.Linear(256, N_labels)
        self.fc2 = nn.Sequential(
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,N_labels)
        )
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        # self.reg_params = self.conv1.parameters()
        # self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        x=x.clamp(-0.4,0.4)
        x0 = x
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        # x = F.relu(self.conv4(x, edge_index))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# class Pyg_Net(torch.nn.Module):
#     def __init__(self):
#         super(Pyg_Net, self).__init__()
#         self.conv1 = GCNConv(N_feats, N_feats, cached=True,
#                              normalize=True)
#         self.conv2 = GCNConv(N_feats, N_feats, cached=True,
#                              normalize=True)
#         # self.conv3 = GCNConv(N_feats, 256, cached=True,
#         #                      normalize=True)
#
#         self.fc1 = nn.Linear(N_feats, 256)
#         self.fc2 = nn.Linear(256, N_labels)
#         # self.fc2 = nn.Sequential(
#         #     nn.Linear(256,128),
#         #     nn.Tanh(),
#         #     nn.BatchNorm1d(128),
#         #     nn.Linear(128,128),
#         #     nn.BatchNorm1d(128),
#         #     nn.Linear(128,N_labels)
#         # )
#         # self.conv1 = ChebConv(data.num_features, 16, K=2)
#         # self.conv2 = ChebConv(16, data.num_features, K=2)
#
#         # self.reg_params = self.conv1.parameters()
#         # self.non_reg_params = self.conv2.parameters()
#
#     def forward(self, x, edge_index):
#         #x, edge_index = data.x, data.edge_index
#         x=x.clamp(-0.4,0.4)
#         x0 = x
#         x = F.relu(self.conv1(x, edge_index))
#         # x = F.dropout(x, training=self.training)
#         x = F.relu(self.conv2(x, edge_index))
#         # x = F.relu(self.conv3(x, edge_index))
#         # x = F.dropout(x, training=self.training)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)



class Pyg_SAGE(torch.nn.Module):
    def __init__(self):
        super(Pyg_SAGE, self).__init__()
        self.conv1 = SAGEConv(N_feats, 200,
                             normalize=True)
        self.conv2 = SAGEConv(200, 128,
                             normalize=True)
        self.conv3 = SAGEConv(128, 128,
                              normalize=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, N_labels)
        )
        # self.fc2 = nn.Linear(256, N_labels)
        # self.conv1 = ChebConv(data.num_afeatures, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        # self.reg_params = self.conv1.parameters()
        # self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        x=x.clamp(-0.4,0.4)
        x0 = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Pyg_GAT(torch.nn.Module):
    def __init__(self):
        super(Pyg_GAT, self).__init__()
        self.conv1 = GATConv(N_feats, N_feats,heads=1)
        self.conv2 = GATConv(N_feats, N_feats,heads=1)
        self.fc1 = nn.Linear(N_feats, 256)
        self.fc2 = nn.Linear(256, N_labels)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        # self.reg_params = self.conv1.parameters()
        # self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        x=x.clamp(-0.4,0.4)
        x0 = x
        x = F.relu(self.conv1(x, edge_index))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        # return x

class Pyg_GIN(torch.nn.Module):
    def __init__(self):
        super(Pyg_GIN, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(N_feats,N_feats),
            nn.ReLU()
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(N_feats,N_feats),
            nn.ReLU()
        ))
        self.fc1 = nn.Linear(N_feats, 256)
        self.fc2 = nn.Linear(256, N_labels)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        # self.reg_params = self.conv1.parameters()
        # self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        x=x.clamp(-0.4,0.4)
        x0 = x
        x = F.relu(self.conv1(x, edge_index))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index)+x0)
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        return x

class Pyg_LEGCN(torch.nn.Module):
    def __init__(self):
        super(Pyg_LEGCN, self).__init__()
        # self.conv1 = GATConv(N_feats, 8, heads=8, dropout=0.6)
        # # On the Pubmed dataset, use heads=8 in conv2.
        # self.conv2 = GATConv(8 * 8, N_labels, heads=1, concat=True,
        #                         dropout=0.6)
        self.conv1 = LEConv(N_feats, 72)
        self.conv2 = LEConv(72, N_labels)

    def forward(self, x, edge_index):
        #x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":

    # 1 basic parameter config
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    use_cuda = 2
    if use_cuda:
        device = torch.device('cuda:' + str(use_cuda))
    else:
        device = torch.device('cpu')

    # 2 read and processing data
    data = Dataset(data_path='/home/zhengyi/')
    # idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    adj, features, labels = data.adj, data.features, data.labels

    adj = sparse_mx_to_torch_sparse_long_tensor(adj)
    # features = sparse_mx_to_torch_sparse_tensor(features)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels - 1)
    # print(torch.min(labels))
    data = Data(x=features, edge_index=adj, y=labels).to(device)
    # gen idx_train, idx_val, idx_test
    _idx = np.arange(len(labels))
    val_size = 0.1
    test_size = 0.8
    train_size = 1 - val_size - test_size
    stratify = labels
    idx_train_and_val, idx_test = train_test_split(_idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    data.train_idx, data.val_idx, data.test_idx = idx_train, idx_val, idx_test

    model = Pyg_Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


    # optimizer = torch.optim.Adam([
    #     dict(params=model.reg_params, weight_decay=5e-4),
    #     dict(params=model.non_reg_params, weight_decay=0)
    # ], lr=0.01)

    # GAT
    # class Pyg_Net(torch.nn.Module):
    #     def __init__(self):
    #         super(Pyg_Net, self).__init__()
    #         self.conv1 = GATConv(N_feats, 8, heads=8, dropout=0.6)
    #         # On the Pubmed dataset, use heads=8 in conv2.
    #         self.conv2 = GATConv(8 * 8, N_labels, heads=1, concat=True,
    #                              dropout=0.6)

    #     def forward(self,data):
    #         #x=data.x
    #         x = F.dropout(data.x, p=0.6, training=self.training)
    #         x = F.elu(self.conv1(x, data.edge_index))
    #         x = F.dropout(x, p=0.6, training=self.training)
    #         x = self.conv2(x, data.edge_index)
    #         return F.log_softmax(x, dim=1)
    # model = Pyg_Net().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(data.x, data.edge_index)[data.train_idx], data.y[data.train_idx]).backward()
        optimizer.step()


    @torch.no_grad()
    def test():
        model.eval()
        logits, accs = model(data.x, data.edge_index), []
        for _, idx in data('train_idx', 'val_idx', 'test_idx'):
            pred = logits[idx].max(1)[1]
            acc = pred.eq(data.y[idx]).sum().item() / idx.shape[0]
            accs.append(acc)
        return accs


    best_val_acc = test_acc = 0
    for epoch in range(1, 1001):
        train()
        train_acc, val_acc, test_acc = test()
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     test_acc = tmp_test_acc
        if epoch < 20 or epoch % 1 == 0:
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, val_acc, test_acc))

    model.cpu()
    torch.save(model.state_dict(), 'model/pyg_net_stage2.pth')