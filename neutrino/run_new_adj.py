#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description    :
@Time    :2020/07/23 17:02:25
@Author    :sam.qi
@Version    :1.0
'''


import sys
import os
import time
from utils import *
import xgboost as xgb


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


def model_to_predict(features, output_path):
    '''
    @Description    :
        5. 特征送入XGBoost 模型训练或预测
        6. 输出最后结果
    @Time    :2020/07/23 17:07:31
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # ML model path
    model_path = "xgboost-v01.model"

    # load model
    param = {}
    param["predictor"] = 'gpu_predictor'
    param["tree_method"] = "gpu_hist"
    param["gpu_id"] = 0
    param["nthread"] = -1
    param["objective"] = 'multi:softmax'

    bst = xgb.Booster(param)
    bst.load_model(model_path)

    print("Finish Model loaded!!，Begin Predict")

    # start predict
    tstart = time.time()
    dpredict = xgb.DMatrix(features)
    result = bst.predict(dpredict)
    res_df = pd.DataFrame(
        {"res": [get_idx_label(int(k)) for k in result.tolist()]})
    tend = time.time()

    # output
    res_df.to_csv(output_path, header=False, index=False)


def predict(adj_path, feature_path, output_path):
    '''
    @Description    :
    流程：
        1. 加载邻接矩阵和特征向量
        2. 从邻接矩阵中找出 degree 为100 的点
        3. 删除 degree = 100 点的边
        4. 所有节点邻居计算平均特征，并和当前特征矩阵合并
        5. 特征送入XGBoost 模型训练或预测
        6. 输出最后结果
    @Time    :2020/07/23 17:02:40
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''
    print("load_adj_feature")
    adj, features = load_adj_feature(adj_path, feature_path)
    print("delete_node_with_100_degree")
    adj = delete_node_with_100_degree(adj)
    dump_pickle(output_path, adj)


if __name__ == "__main__":
    args = sys.argv
    predict(args[1], args[2], args[3])
