#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description    :
@Time    :2020/07/23 17:00:03
@Author    :sam.qi
@Version    :1.0
'''


import numpy as np
import matplotlib.pyplot as plt
import pickle
from xgboost import plot_importance
from scipy.sparse import coo_matrix
import datetime
from scipy.sparse import diags
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import torch


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_numpy(path, allow_pickle=False):
    '''
    @说明    : 加载Numpy 类型文件
    @时间    :2020/06/05 10:42:10
    @作者    :sam.qi
    '''

    obj = np.load(path, allow_pickle=allow_pickle)

    return obj


def dump_pickle(path, obj, delete=True):
    '''
    @说明    : 保存 pickle 文件
    @时间    :2020/06/05 10:42:36
    @作者    :sam.qi
    '''

    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def save_npy(path, obj):
    '''
    @说明    : 以npy 格式保存对象
    @时间    :2020/06/05 10:42:57
    @作者    :sam.qi
    '''

    np.save(path, obj)


def show_import_feature(bst):
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_importance(bst,
                    height=0.5,
                    ax=ax,
                    max_num_features=6)
    plt.show()
    plt.savefig("xgboost-feature-importance.png")
    importance = bst.get_fscore()

    keys = list(sorted(importance.items(),
                       key=lambda kv: kv[1], reverse=True))[:6]
    print(keys)
#     fileutiles.dump_pickle("data/importance.pkl",
#                            importance)


def draw_images(results):
    # retrieve performance metrics
    epochs = len(results['eval']['merror'])
    x_axis = range(0, epochs)

    plt.figure(figsize=(100, 20))
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(x_axis, results['eval']['mlogloss'], label='Test')
    ax[0].plot(x_axis, results['train']['mlogloss'], label='Train')
    ax[0].legend()
    # ax[0].title('XGBoost Log Loss')

    # plot classification error
    for result in results:
        ax[1].plot(x_axis, results['eval']['merror'], label='Test')
        ax[1].plot(x_axis, results['train']['merror'], label='Train')
    ax[1].legend()
    # ax[1].title('XGBoost Classification Error')
    plt.show()
    plt.savefig("train_process.png")


def search_node_by_degree(adj, degree=100):
    start = datetime.datetime.now()
    node_degree = adj.sum(axis=0)
    node_degree = np.array(node_degree.tolist())
    target_idx = list(np.where(node_degree == degree)[1])
    end = datetime.datetime.now()
    time_span = (end-start).seconds
    print(str.format("+++++++ Feature [Search Node Degree] :{}s", time_span))
    return target_idx


def average_node_feature(adj, features):
    neighbor_features = adj.dot(features)
    node_degree = adj.sum(axis=0)
    node_degree = np.array(node_degree.tolist())
    node_degree = node_degree.T

    avg_neighbor_feature = neighbor_features / node_degree
    return avg_neighbor_feature


def reset_sparse(adj, clear_nodes):
    '''
    @Description    : 清除邻接矩阵节点
        1. 构建对角阵 A
        2. 将 clear_nodes 对应位置致0
        3. 对角阵A 乘以 adj 清空 adj 对应行
        4. adj 转置
        5. 对角阵A 乘以 adj 清空 adj 对应列
        5. adj 转置，回复原来顺序
    @Time    :2020/07/23 18:53:21
    @Author    :sam.qi
    @Param    :
    @Return    :

    '''
    start = datetime.datetime.now()

    # 构建对角阵
    shape = adj.shape
    row_size = shape[0]
    diag_row = list(range(row_size))
    diag_col = list(range(row_size))
    diag_data = [1]*row_size

    for n_idx in clear_nodes:
        diag_data[n_idx] = 0

    diag_M = coo_matrix((diag_data, (diag_row, diag_col)), shape=shape)

    # 清空行和列
    r_clear_adj = diag_M.dot(adj)
    r_c_clear_adj = diag_M.dot(r_clear_adj.T)

    # 恢复
    new_adj = r_c_clear_adj.T
    new_adj = new_adj.tocsr()

    end = datetime.datetime.now()
    time_span = (end-start).seconds
    print(str.format("+++++++ Feature [Reset Sparse] :{}s", time_span))
    return new_adj


def deal_adj(adj):
    '''
    @Description    :
        1. 构建对称矩阵
        2. 重置大于1 的值为1 
        3. 将对角线清空
        4. 去除 0 值
    @Time    :2020/07/24 11:56:15
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    start = datetime.datetime.now()

    adj = adj + adj.T

    stage1 = datetime.datetime.now()
    time_span = (stage1-start).seconds
    print(str.format("+++++++ GCN Deal Adj[Symmetry] :{}s", time_span))

    # adj = adj.tolil()
    adj[adj > 1] = 1
    # whether to set diag=0?
    stage2 = datetime.datetime.now()
    time_span = (stage2-stage1).seconds
    print(str.format("+++++++ GCN Deal Adj[Reset Value To 1] :{}s", time_span))

    # adj.setdiag(0)
    adj = adj - diags(adj.diagonal())

    stage3 = datetime.datetime.now()
    time_span = (stage3-stage2).seconds
    print(str.format("+++++++ GCN Deal Adj[To CSR] :{}s", time_span))

    adj.eliminate_zeros()

    stage4 = datetime.datetime.now()
    time_span = (stage4-stage3).seconds
    print(str.format("+++++++ GCN Deal Adj[Eliminate Zeros] :{}s", time_span))
    return adj


def create_kford_train_set(stratify, n_splits=5, seed=24):
    '''
    @Description    : 构建 KFord 交叉验证

    @Time    :2020/07/24 16:26:43
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    val_size = 0.2

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(stratify))

    # K-折
    kf = KFold(n_splits=n_splits)

    return list(kf.split(idx))


def accuracy(output, labels):
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


