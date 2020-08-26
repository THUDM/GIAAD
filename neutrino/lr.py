import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from utils import *


class LR(nn.Module):

    def __init__(self, nfeat, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device=None):

        super(LR, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.linear = nn.Linear(nfeat, nclass)

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
        self.features = None

    def forward(self, x):
        '''
            adj: normalized adjacency matrix
        '''
        x = F.relu(self.linear(x))

        return F.log_softmax(x, dim=1)

    def fit(self, ftrain_features, train_label, val_features, val_label, patience=10, verbose=True, train_iters=1000):
        self.features = ftrain_features
        self.labels = train_label

        self.val_features = val_features
        self.val_labels = val_label

        self._train_with_early_stopping(train_iters, patience, verbose)

    def predict(self, features=None):
        '''By default, inputs are unnormalized data'''

        self.eval()
        self.forward(features)

    def _train_with_early_stopping(self, train_iters, patience, verbose):
        if verbose:
            print('=== training Logist Classification model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100
        best_acc = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features)
            loss_train = F.nll_loss(output, self.labels)
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.val_features)

            loss_val = F.nll_loss(output, self.val_labels)
            acc_val = accuracy(output, self.val_labels)

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}, validate loss:{}, validate acc:{}'.format(
                    i, loss_train.item(), loss_val.item(), acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                best_acc = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:

            print('=== early stopping at {0}, loss_val = {1}, accurate = {2} ==='.format(
                i, best_loss_val, best_acc))
        self.load_state_dict(weights)
