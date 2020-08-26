import sys; sys.path.append('../')
import os
import pickle
import zipfile
import unittest

import numpy as np
from scipy.sparse import hstack, vstack
from sklearn.model_selection import train_test_split

import torch; print(torch.__version__)
import torch.nn.functional as F
#from torch.nn import Linear, Sequential, ReLU, Embedding
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data

from models import GCN, test
#from d_attack.models import GCN, test

SEED = 0
ADJ_SIZE = 593486
TEST_SIZE = 50000
TRAIN_SIZE = ADJ_SIZE-TEST_SIZE
FEATURE_DIM = 100


class Test1Input(unittest.TestCase):

    def setUp(self):
        # zipを解凍
        with zipfile.ZipFile("./submit/submit.zip") as zf:
            zf.extractall("./submit/tmp/")
        # ファイルをリスト化
        self.flist = sorted(os.listdir("./submit/tmp/"))
        # 行列取り出し
        self.arr_adj = np.load('./submit/tmp/'+self.flist[0], allow_pickle=True)
        self.arr_feat = np.load('./submit/tmp/'+self.flist[1])
        # ノード数とエッジ数
        self.k = self.arr_adj.shape[0]
        self.e = self.arr_adj.count_nonzero()/2

    def test_firstcheck(self):
        print("\n")
        print("ファイル名:\t", self.flist)
        print("ノード数:\t", self.k)
        print("エッジ数:\t", self.e)

    def test_fname(self):
        """ファイル名が正しいかどうか確認"""
        self.assertEqual(self.flist[0], "adj.pkl")
        self.assertEqual(self.flist[1], "feature.npy")

    def test_nodecount(self):
        """追加されたノードが500以下であることを確認する"""
        self.assertLessEqual(self.arr_adj.shape[0], 500)

    def test_edgecount(self):
        """1ノードあたりのエッジ数が100以下であることを判定する"""
        self.assertLessEqual(self.arr_adj.getnnz(axis=1).max(), 100)

    def test_symmetric(self):
        """追加ノードの隣接行列が対称行列になっているか判定する"""
        self.assertTrue((self.arr_adj[:, ADJ_SIZE:]-self.arr_adj[:, ADJ_SIZE:].T).nnz==0)

    def test_features_matrix_size(self):
        """提出する特徴量行列のサイズを確認する"""
        self.assertTrue((self.arr_feat.shape == (self.k, 100)))

    def test_features_matrix_values(self):
        """提出する特徴量行列の値を確認する"""
        self.assertLessEqual(self.arr_feat.max(), 100)
        self.assertGreaterEqual(self.arr_feat.min(), -100)



class Test2Attack(unittest.TestCase):
    def setUp(self):
        print("\n### Attack Test ###")
        self.path = "./pyg_model/"
        self.fname_model = "20200601_GCN_model.pkl"
        self.fname_params = "20200601_GCN_params.pkl"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def test_GCN(self):
        # データ読み込み
        data_clean, data_attacked = load_test_data("../../../../mltrack2_data/", kind="lgb")

        # 学習済みのモデル読み込み
        model, params = load_optimized_model(self.path,
                                            data_clean,
                                            self.device,
                                            self.fname_model, 
                                            self.fname_params)
        print("\n")
        print("Hyper Parameter: ", params)
        print("\n")
        print("Clean Data: ", data_clean)
        print("\n")
        print("Attacked Data: ", data_attacked)
        print("\n")
        
        # ローカルの精度検証
        before_acc, before_loss = test(model, data_clean, self.device)
        after_acc, after_loss = test(model, data_attacked, self.device)
        
        # 精度出力
        print(f'Before Acc : {before_acc:.4f}, Before Loss : {before_loss:.4f}')
        print(f'After  Acc : {after_acc:.4f}, After  Loss :  {after_loss:.4f}')
        
        print(f'Acc Diff   : {before_acc-after_acc}')
        pass
    
    def tearDown(self):
        print("### DONE ###")
        pass


def load_optimized_model(path, data, device, fname_model, fname_params):
    """学習済みのモデルを読み込んでPyGモデルとパラメータを返す"""
    with open(path+fname_params, "rb") as f:
        params = pickle.load(f)

    model = GCN(data.num_node_features,
                data.num_class,
                params['hidden'],
                params['dropout_rate']).to(device)

    model.load_state_dict(torch.load(path+fname_model, map_location=device))

    return model, params


def load_test_data(path, kind):
    """
    攻撃前のデータと攻撃後のデータを返す関数
    
    data_clean: 改竄されていない元データ
    data_attacked: submit.zipのデータを追加した攻撃後のデータ
    
    Parameters
    --------
    path: mltrack2のデータセットの格納場所
    
    kind: 
        "local" 学習時に用意したvaidationセットで精度比較
        "lgb"   LightGBM(acc 0.857458)

    Returns
    --------
    pyg Data
        data_clean, data_attacked

    """
    
    def transform(adj, attr, labels):
        # PytorchのTensor型に変換する
        edge_index, edge_attr = from_scipy_sparse_matrix(adj)
        x = torch.tensor(attr, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)
        # PygのDataクラスを作成
        data = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_attr)
        data.num_class = len(np.unique(y))
        return data
    
    def masking(indices, num_nodes):
        """与えられたインデックスの部分だけ値が1になっているリストを作成"""
        masked = torch.zeros(num_nodes, dtype=torch.bool)
        masked[indices] = 1
        return masked


    # 元データの読み込み
    adj_matrix = np.load(path+"experimental_adj.pkl", allow_pickle=True)
    attr_matrix = np.load(path+"experimental_features.pkl", allow_pickle=True)
    labels = np.load(path+"experimental_train.pkl", allow_pickle=True)
    labels_lgb = np.load(path+"20200530_testlabel_lgb.pkl", allow_pickle=True)
    labels_assumed = np.append(labels, labels_lgb)

    # 攻撃データの読み込み
    dir = os.path.dirname(os.path.abspath(__file__))
    with zipfile.ZipFile(dir+"/submit/submit.zip") as zf:
        zf.extractall(dir+"/submit/tmp/")
    # 行列取り出し
    arr_adj = np.load(dir+'/submit/tmp/adj.pkl', allow_pickle=True)
    arr_feat = np.load(dir+'/submit/tmp/feature.npy')

    # 攻撃データを隣接行列にマージ
    # arr_adjを転置して下(500*500)を除いた部分をAにh_stack -> (593486x593986)
    arr_upper = hstack([adj_matrix, arr_adj.T[:-500,:]])
    # arr_adj(500x593986)をv_stackする -> (593986x593986)
    adj_all = vstack([arr_upper, arr_adj])
    # arr_feat(500×100)をv_stackする　-> (593986×100)
    feat_all = vstack([attr_matrix, arr_feat]).toarray()
    labels_all = np.append(labels_assumed, np.zeros(500))

    data_clean = transform(adj_matrix, attr_matrix, np.append(labels, labels_lgb))
    data_attacked = transform(adj_all, feat_all, labels_all)
    
    # テスト用のインデックス、マスキングを設定
    if kind=="local":
        _, data_clean.test_indices = train_test_split(np.arange(TRAIN_SIZE),
                                                      test_size=0.2,
                                                      random_state=SEED)
    elif kind=="lgb":
        data_clean.test_indices = np.arange(TRAIN_SIZE, ADJ_SIZE)
    else:
        pass
    data_clean.test_mask = masking(data_clean.test_indices, ADJ_SIZE)
    data_attacked.test_indices = data_clean.test_indices
    data_attacked.test_mask = masking(data_attacked.test_indices, data_attacked.num_nodes)

    return data_clean, data_attacked


if __name__ == '__main__':
    unittest.main()
