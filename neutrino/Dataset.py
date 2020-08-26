import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class Dataset():

    def __init__(self, data_path, seed=None, require_mask=False):

        self.seed = seed
        self.data_path = data_path
        self.adj, self.features, self.labels = self.load_data()
        # self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test(
        #     stratify=self.labels)
        self.k_ford_idx = self.get_train_val_test(
            stratify=self.labels)

    def get_train_val_test(self, stratify, n_splits=5):
        val_size = 0.1
        test_size = 0.8

        if self.seed is not None:
            np.random.seed(self.seed)

        idx = np.arange(len(self.labels))

        # K-折
        kf = KFold(n_splits=n_splits)

        train_size = 1 - val_size - test_size
        # idx_train_and_val, idx_test = train_test_split(idx,
        #                                                random_state=None,
        #                                                train_size=train_size + val_size,
        #                                                test_size=test_size,
        #                                                stratify=stratify)

        k_ford_index = []
        for idx_train_and_val, idx_test in kf.split(idx):

            if self.labels is not None:
                sub_stratify = stratify[idx_train_and_val]

            idx_train, idx_val = train_test_split(idx_train_and_val,
                                                  random_state=None,
                                                  train_size=(
                                                      train_size / (train_size + val_size)),
                                                  test_size=(
                                                      val_size / (train_size + val_size)),
                                                  stratify=sub_stratify)
            k_ford_index.append((idx_train, idx_val, idx_test))
        return k_ford_index

    def load_data(self):
        adj, features, labels = self.get_adj()
        return adj, features, labels

    def deal_adj(self, adj):
        adj = adj + adj.T
        # adj = adj.tolil()
        adj[adj > 1] = 1
        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()
        return adj

    def get_adj(self):
        adj, features, labels = self.load_npz()
        self.raw_adj = adj
        adj = self.deal_adj(adj)
        # assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        # assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"
        return adj, features, labels

    def load_npz(self, is_sparse=True):

        # [0:543486, 0:543486]
        adj = np.load(self.data_path['adj'], allow_pickle=True)
        features = np.load(
            self.data_path['feat'], allow_pickle=True)  # [0:543486]
        labels = np.load(self.data_path['label'], allow_pickle=True)

        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels
