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
import datetime
from utilsr import *
import xgboost as xgb
import pandas as pd
from labelconvert import get_idx_label
import torch
import numpy as np
import GCN
import gcnpredict
from common import *


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

    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # ML model path
    model_path = "Neutrino/xgboost-v01.model"

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
    
   
    return result
 


def gcn_model_to_predict(adj, features):
    '''
    @Description    :
        5. 特征送入XGBoost 模型训练或预测
        6. 输出最后结果
    @Time    :2020/07/23 17:07:31
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''
    start = datetime.datetime.now()
    new_adj = deal_adj(adj)
    stage1 = datetime.datetime.now()
    spend_time = (stage1-start).seconds
    print(str.format("-------GCN [Deal Adj] : {}s", spend_time))

    result = gcnpredict.predict(new_adj, features)
    return result
    

    

def predict(adj, features):
    
    adj = delete_node_with_100_degree(adj)
    
    return gcn_model_to_predict(adj, features)
    
