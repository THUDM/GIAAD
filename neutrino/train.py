#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description    :
    流程：
        1. 加载邻接矩阵和特征向量
        2. 所有节点邻居计算平均特征，并和当前特征矩阵合并
        3. 特征送入XGBoost 模型训练或预测
        4. 保存模型
@Time    :2020/07/23 17:30:36
@Author    :sam.qi
@Version    :1.0
'''

import sys
from utils import *
from common import *
from xgboostmodel import xgboost_model_train
from labelconvert import get_label_idx
import os


def load_labels(labels_path):
    '''
    @Description    :
    @Time    :2020/07/23 17:41:55
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    labels = list(load_numpy(labels_path))
    new_labels = [get_label_idx(l) for l in labels]
    return new_labels


def split_dataset(features, labels):
    '''
    @Description    :
    拆分数据集
    @Time    :2020/07/23 17:40:37
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    # 数据类型转换
    features = np.array(features)
    labels = np.array(labels)

    validate_index_path = "validate_index.pkl"

    # 生成训练和验证索引

    sample_size = len(labels)
    all_idx = set(range(sample_size))
    validate_idx = load_pickle(validate_index_path)
    train_idx = all_idx - set(validate_idx)

    train_idx = list(train_idx)
    validate_idx = list(validate_idx)

    # 生成训练和验证特征与标签
    train_feature = features[train_idx]
    train_labels = labels[train_idx]

    validate_feature = features[validate_idx]
    validate_labels = labels[validate_idx]

    return train_feature, train_labels, validate_feature, validate_labels


def train(adj_path, features_path, labels_path):
    '''
    @Description    :
    @Time    :2020/07/23 17:33:46
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''
    adj, features = load_adj_feature(adj_path, features_path)
    final_features = adjust_feature(adj, features)
    labels = load_labels(labels_path)

    print(final_features.shape)
    train_feature, train_labels, validate_feature, validate_labels = split_dataset(
        final_features, labels)
    xgboost_model_train(train_feature, train_labels,
                        validate_feature, validate_labels)


if __name__ == "__main__":
    args = sys.argv
    train(args[1], args[2], args[3])
