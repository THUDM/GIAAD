#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description    :
@Time    :2020/07/23 17:33:19
@Author    :sam.qi
@Version    :1.0
'''

from utilsr import *


def load_adj_feature(adj_path, feature_path):
    '''
    @Description    : 
        1.加载邻接矩阵和特征向量
    @Time    :2020/07/23 17:04:21
    @Author    :sam.qi
    @Param    :
    @Return    : adj, features
    '''

    adj = load_pickle(adj_path)
    features = load_numpy(feature_path)
    return adj, features


def adjust_feature(adj, features):
    '''
    @Description    :
        4.所有节点邻居计算平均特征，并和当前特征矩阵合并
    @Time    :2020/07/23 17:06:15
    @Author    :sam.qi
    @Param    :
    @Return    : final_features
    '''

    avg_features = average_node_feature(adj, features)
    final_features = np.hstack((features, avg_features))
    return final_features


def delete_node_with_100_degree(adj):
    '''
    @Description    :
        2. 从邻接矩阵中找出 degree 为100 的点
        3. 删除 degree = 100 点的边
    @Time    :2020/07/23 17:05:23
    @Author    :sam.qi
    @Param    :
    @Return    : adj
    '''
    node_degree_100 = search_node_by_degree(adj)
    new_adj = reset_sparse(adj, node_degree_100)
    return new_adj
