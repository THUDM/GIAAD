import pickle
import numpy as np
import scipy.sparse as sp
import torch


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_nodes(features, adj, labels, idx_train, target_node, n_added=1, n_perturbations=10):
    print('number of pertubations: %s' % n_perturbations)
    N = adj.shape[0]
    D = features.shape[1]
    modified_adj = reshape_mx(adj, shape=(N+n_added, N+n_added))
    modified_features = self.reshape_mx(features, shape=(N+n_added, D))

    diff_labels = [l for l in range(
        labels.max()+1) if l != labels[target_node]]
    diff_labels = np.random.permutation(diff_labels)
    possible_nodes = [x for x in idx_train if labels[x] == diff_labels[0]]

    return modified_adj, modified_features


def generate_injected_features(features, n_added):
    # TODO not sure how to generate features of injected nodes
    features = features.tolil()
    # avg = np.tile(features.mean(0), (n_added, 1))
    # features[-n_added: ] = avg + np.random.normal(0, 1, (n_added, features.shape[1]))
    features[-n_added:] = np.tile(0, (n_added, 1))
    return features


def injecting_nodes(data):
    '''
        injecting nodes to adj, features, and assign labels to the injected nodes
    '''
    adj, features, labels = data.raw_adj, data.features, data.labels
    # features = normalize_feature(features)
    N = adj.shape[0]
    D = features.shape[1]
    n_added = 500
    print(N)
    print('number of injected nodes: %s' % n_added)

    data.adj = reshape_mx(adj, shape=(N+n_added, N+n_added))
    enlarged_features = reshape_mx(features, shape=(N+n_added, D))
    data.features = generate_injected_features(enlarged_features, n_added)
    # data.features = normalize_feature(data.features)

    injected_labels = np.random.choice(labels.max()+1, n_added)
    # data.labels = np.hstack((labels, injected_labels))


def normalize_adj(adj):
    """
    A_hat = D_rev*(D+A)*D_rev
    A_hat = D_rev*(D-A)*D_rev
    todo: many ways to try!
    """
    A_hat = adj
    degree = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = sp.diags(degree)
    degree_reverse = degree ** (-1 / 2)
    degree_reverse[np.isinf(degree_reverse)] = 0.
    D_hat_reverse = sp.diags(degree_reverse)
    adj = D_hat_reverse * (D_hat + A_hat) * D_hat_reverse
    return adj


def reshape_mx(mx, shape):
    indices = mx.nonzero()
    return sp.csr_matrix((mx.data, (indices[0], indices[1])), shape=shape)
