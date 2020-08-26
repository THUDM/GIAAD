import sys; sys.path.append('./d_attack/')

import numpy as np
import scipy


import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from d_attack import const


def load_pkl(fdict, with_test=True):
    """隣接行列、特徴量行列、ラベルを返す関数"""
    adj = np.load(fdict["adj"], allow_pickle=True)
    attr = np.load(fdict["feature"], allow_pickle=True)
    labels = np.random.randint(0, 19, (609574, ))
    #labels = np.zeros((adj.shape[0], ))
    return adj, attr, labels


def load_pyg_data(adj,attr):
    """pygデータを返す関数"""
    #adj, attr, labels = load_pkl(fdict, with_test=True)
    
    # preprocess 特徴量
    empty_features_inx = np.where(np.all(attr==0, axis=1))
    attr, remove_list = preprocess_feat(attr)

    # preprocess 隣接行列
    adj = preprocess_adj(adj, remove_list)
    
    # preprocess ラベル
    #labels = preprocess_labels(labels, reverse=False)
    
    # PytorchのTensor型に変換する
    edge_index, edge_attr = from_scipy_sparse_matrix(adj)
    x = torch.tensor(attr, dtype=torch.float)
    #y = torch.tensor(labels, dtype=torch.long)
    
    data = Data(x=x,
     #           y=y,
                edge_index=edge_index,
                edge_weight=edge_attr,
                num_class=18)

    # 特徴量が0のインデックスを取得
    data.empty_features_inx = empty_features_inx
   # print(f"Detectded ALL 0 features !", data.empty_features_inx)

    return data


def preprocess_adj(adj_matrix, remove_list):
    """隣接行列の対称化&重みを0に変換"""
    if(set(adj_matrix.data)!={1}):
        #print(f"Adj Matrix is Weighted! {set(adj_matrix.data)} Change Weight to 0...")
        # 重みが1以外のものはダミーなので0に変換
        # timeoutの可能性があるので[ !=1]での比較は行わない
        adj_matrix[adj_matrix > 1] = 0
        adj_matrix[adj_matrix < 0] = 0
        # 念のため隣接行列を対称化
        #adj_matrix = adj_matrix + adj_matrix.T
        #adj_matrix[adj_matrix > 1] = 1
    else:
        ww=0
        #print(set(adj_matrix.data), "OK(^_^)! Adj Matrix is Unweighted.")

    # 次数が90~100のノードを除去
    #print("Canselling Adversarial Nodes...")
    deg = np.array(adj_matrix.getnnz(axis=1))
    # 残したいインデックスを絞り込む
    filter = (deg<90)|(deg>100)
    # remove_listをfilterから除去
    #print("Cancelling Nodes by Features Remove List...", remove_list)
    filter[remove_list]=False
    I = scipy.sparse.diags(1*filter).tocsr()
    adj_matrix = I.T*adj_matrix*I
    
    return adj_matrix


def preprocess_feat(attr_matrix):
    """特徴量行列の最大値最小値を基準に攻撃ノードを判定＆除去する"""
    max_list, min_list = np.max(attr_matrix, axis=1), np.min(attr_matrix, axis=1)
    remove_list = np.concatenate([np.where(max_list > const.FEATURE_MAX)[0],
                                  np.where(min_list<const.FEATURE_MIN)[0]])
    #print(f"Detected adversarial feature: {len(remove_list)}")
    attr_matrix[remove_list]=0

    #print("Scaling values of attr_matrix...")
    attr_matrix = scipy.stats.zscore(attr_matrix, axis=None)
    return attr_matrix, remove_list


def preprocess_labels(labels, reverse=False):
    """ラベルを振り直す関数"""
    original = [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    conversion = np.arange(len(original))
    if reverse == True:
        dic = dict(zip(conversion, original))
      #  print("Reverse: ", dic)
    else:
        dic = dict(zip(original, conversion))
        #print("Normal: ", dic)
    return np.array(list(map(dic.get, labels)))
