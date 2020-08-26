#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description    :
@Time    :2020/07/24 16:05:01
@Author    :sam.qi
@Version    :1.0
'''

from utils import *
from common import *
import gcnpredict
import numpy as np
from multiprocessing import Process, Queue
from sklearn.linear_model import LogisticRegression
from lr import LR
import torch

################################################################################################################
adj_path = "/data1/qyf/data//final_new_adj.pkl"
feature_path = "/data1/qyf/data//feature_formal_stage.npy"
label_path = "/data1/qyf/data//train_labels_formal_stage.npy"

################################################################################################################


def compute_gcn_feature(adj, features):
    '''
    @Description    :
    @Time    :2020/07/24 16:19:36
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    new_adj = deal_adj(adj)
    gcn_feature = gcnpredict.predict(new_adj, features, output_prob=True)
    new_features = np.hstack((features, gcn_feature))
    return new_features


def craete_train_set():
    '''
    @Description    : 
    @Time    :2020/07/24 16:25:17
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''
    labels = load_numpy(label_path)
    kford_ds = create_kford_train_set(labels)

    return kford_ds, labels


def trainLR(model_id, train_features, train_label, val_features, val_label):
    '''
    @Description    :
    @Time    :2020/07/24 16:23:41
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''
    print(str.format("LR-{} start training", model_id))

    use_cuda = True
    if use_cuda:
        device = torch.device(str.format('cuda:{}', model_id)
                              ) if torch.cuda.is_available() else 'cpu'
    else:
        device = torch.device('cpu')

    clf = LR(120, 20, device=device)
    clf.to(device)
    train_features = train_features.to(device)
    train_label = train_label.to(device)
    val_features = val_features.to(device)
    val_label = val_label.to(device)
    clf.fit(train_features, train_label,
            val_features, val_label, train_iters=2000)


def mul_trainlr(train_set, features, labels):
    '''
    @Description    :
    @Time    :2020/07/24 16:37:08
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    pss = []
    for i, ds_idx in enumerate(train_set):
        idx_train, idx_val = ds_idx
        idx_train = list(idx_train)
        idx_val = list(idx_val)

        train_features = features[idx_train]
        print("features")
        train_label = labels[idx_train]
        print("train_label")
        val_features = features[idx_val]
        print('val_features')
        val_label = labels[idx_val]

        trainLR(i, train_features, train_label, val_features, val_label)
        # p = Process(target=trainLR, args=[
        #             i, idx_train, idx_val])
        # p.daemon = True
        # p.start()
        # pss.append(p)

    for p in pss:
        p.join()

    print("K-Ford LR Train Finished....")


def train():
    '''
    @Description    :
        1. 加载数据
        2. 计算gcn 特征
        3. 特征拼接
        4. 定义模型开始训练
        5. 保存模型
    @Time    :2020/07/24 16:07:07
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    print("load_adj_feature")
    adj, features = load_adj_feature(adj_path, feature_path)

    print("delete_node_with_100_degree")
    adj = delete_node_with_100_degree(adj)

    print("compute GCN feature and vstack")
    features = compute_gcn_feature(adj, features)

    print("Create Train Set")
    train_set, labels = craete_train_set()

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    print("Train LR")
    mul_trainlr(train_set, features, labels)

################################################################################################################


if __name__ == "__main__":
    train()
