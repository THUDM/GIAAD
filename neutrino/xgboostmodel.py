#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description    :
@Time    :2020/07/23 17:37:37
@Author    :sam.qi
@Version    :1.0
'''

import xgboost as xgb
from sklearn.metrics import f1_score
from utils import *


def _param():
    '''
    @Description    :
    @Time    :2020/07/23 17:37:33
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    # 定义参数
    param = {}

    # booster paramss
    param["n_estimator"] = 1000
    param['learning_rate'] = 0.04
    param['max_depth'] = 2

    param['random_state'] = 27

    param["subsample"] = 0.8
    param["colsample_btree"] = 0.8

    param['num_class'] = 18

    # learning params
    param['objective'] = 'multi:softmax'
    param['eval_metric'] = ['mlogloss', 'merror']
    param["predictor"] = 'gpu_predictor'
    param["tree_method"] = "gpu_hist"
    param["gpu_id"] = 7

    return param


def _compute_f1(y_true, y_pred):
    '''
    @Description    : compute f1 score
    @Time    :2020/06/05 11:29:38
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    score = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    return score


def xgboost_model_train(train_feature, train_labels, validate_feature, validate_labels):
    '''
    @Description    :
    @Time    :2020/07/23 17:37:31
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    param = _param()
    num_round = 5000
    cpu_res = {}

    # 定义训练和测试集合
    dtrain = xgb.DMatrix(train_feature, label=train_labels)
    dtest = xgb.DMatrix(validate_feature, label=validate_labels)

    # 定义评估对象
    evallist = [(dtrain, 'train'), (dtest, 'eval')]

    # 开始训练
    bst = xgb.train(param, dtrain, num_round,
                    evals=evallist, evals_result=cpu_res, early_stopping_rounds=10)

    # 保存模型
    bst.save_model('xgboost-v01.model')

    # 测试集评估
    y_pred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

    # 计算指标
    score_value = _compute_f1(validate_labels, y_pred)
    print("F1 Score: %.5f%%" % score_value)

    show_import_feature(bst)
    draw_images(cpu_res)
