"""
KDDCUP2020 MLTrack2
https://www.biendata.xyz/competition/kddcup_2020/

Author: NTT DOCOMO LABS
License: MIT
"""
import gc
import pickle
import zipfile

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix

from d_attack import const
from d_attack.models import GCN, test
from d_attack.utils import load_optimized_model, test_adjacent_matrix, test_features_matrix, check_symmetric


class BaseAttacker():
    """
    Attackerで使うデータやラベルの準備を行うクラス

    Attributes
    ----------
    data: Pytorch Geometric Dataクラス


    """

    def __init__(self, data):
        # Dataオブジェクト→隣接行列、特徴量行列、ラベル
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.A = to_scipy_sparse_matrix(data.edge_index, data.edge_weight).tocsr()
        self.X = data.x.cpu().numpy()
        self.labels = data.y.cpu().numpy()
        self.k = 500
        self.e = 100

    def get_lgb_labels(self, fname):
        """LightGBMでテストラベルを推測する"""
        labels_lgb = np.load(const.DATA_PATH+fname, allow_pickle=True)
        return labels_lgb
    
    def submit(self, extended_adj, extended_features, path):
        """submit用のファイルを作成する"""
        # データ保存
        pickle.dump(extended_adj, open("adj.pkl", "wb"))
        np.save("feature.npy", extended_features)
        # 提出ファイル作成
        with zipfile.ZipFile(path+'submit.zip', 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
            new_zip.write(path+'adj.pkl')
            new_zip.write(path+'feature.npy')
        print("Done!!😄")

    def __del__(self):
        pass

        

class RandomAttacker(BaseAttacker):
    """
    隣接行列と特徴量行列をRandomに攻撃する
    
    Notes
    -----
    data
    """
    
    def __init__(self, data):
        super().__init__(data)
        # テストノードの次数を計算
        self.degrees = self.A.getnnz(axis=1)[const.TRAINSIZE:const.ADJSIZE]
        # テストノードの推測ラベルを読み込む
        self.label_lgb = np.load("../../mltrack2_data/20200530_testlabel_lgb.pkl", allow_pickle=True)


    def get_average_features(self, n):
        """nで指定したラベルの平均特徴量を返す"""
        df = pd.DataFrame(self.X[:const.TRAINSIZE])
        df["label"] = self.labels[:const.TRAINSIZE]
        df_mean = df.groupby("label").mean()
        return df_mean.loc[n]


    def stratified_choice(self):
        unique_node, label_count = np.unique(self.labels_lgb, return_counts=True)
        # total 100にならないので決め打ちで配分を決める
        stratified = np.array([14,  3,  7, 23,  3,  9,  3,  3,  3,  3,  3,  4,  6,  3,  3,  3,  3, 4])

        target_indices = np.array([])
        for n in np.arange(18):
            target_inx = np.where(self.labels_lgb==n)[0]
            target_indices = np.append(target_indices, np.random.choice(target_inx, stratified[n], replace=False))
        return target_indices.astype(int)
    
    
    def generate_adj(self, kind="random", n=3):
        """テストデータに対して集中的にエッジを張るような隣接行列を作る"""
        
        A, k, e = self.A, self.k, self.e
        
        # k*kの隣接行列を作る（新規ノード同士は接続させない）
        arr_right = np.zeros((k,k))
        csr_right = csr_matrix(arr_right)

        # k*adjsizeの行列を作る
        arr_left = np.zeros((k, const.ADJSIZE))
        if kind=="random":
            target_indices = np.arange(const.TRAINSIZE, const.ADJSIZE)
        elif kind=="low_degree":
            # 次数が5以下のノードのインデックスを抽出
            target_indices = np.where(self.degrees<=26)[0] + const.TRAINSIZE
        elif kind=="high_degree":
            # 次数が20以上のノードのインデックスを抽出
            target_indices = np.where(self.degrees>=20)[0] + const.TRAINSIZE
        elif kind=="target":
            # 指定したラベルのノードのインデックスを抽出
            target_indices = np.where(self.label_lgb==n)[0] + const.TRAINSIZE
        elif kind=="exclusion":
            # 指定したラベルのノード"以外"のインデックスを抽出
            target_indices = np.where(self.label_lgb!=n)[0] + const.TRAINSIZE
        else:
            target_indices = np.arange(const.ADJSIZE)

        rand_index = np.random.choice(target_indices, len(target_indices), replace=False)

        for v in arr_left:
            if kind=="stratified":
                # 層化抽出する
                v[stratified_choice(degrees, labels_lgb)]=const.ATTACK_VALUE
            else:
                # ランダムにe個選ぶ
                #v[np.random.choice(target_indices, e, replace=False)]=99999
                # 網羅的にe個選ぶ
                v[rand_index[:100]] = const.ATTACK_VALUE
                rand_index = rand_index[100:]

        csr_left = csr_matrix(arr_left)

        # 連結する
        extended_adj = hstack([csr_left, csr_right], format='csr')

        # 生成した隣接行列のチェック
        test_adjacent_matrix(extended_adj, const.ADJSIZE)
        return extended_adj

    
    def generate_features(self, kind="random", n=3):
        """kindに応じて攻撃用の特徴量を生成する"""
        
        if kind=="100":
            extended_features = 100*np.ones((self.k, self.e))
        elif kind=="100/-100":
            arr=np.append(np.ones(25000)*self.e, np.ones(25000)*(-self.e))
            np.random.shuffle(arr)
            extended_features = arr.reshape((self.k, self.e))
        elif kind=="zero":
            extended_features = np.zeros((self.k, self.e))
        elif kind=="gauss":
            mu, sigma = 0, 1
            extended_features = np.random.normal(mu, sigma, (self.k, self.e))
        elif kind=="random":
            extended_features = np.random.rand(self.k, self.e)
        elif kind=="target":
            array = get_average_features(self.X, self.labels, n).values
            extended_features = np.tile(array, (self.k, 1))
        elif kind=="target_inverse":
            # 符号を反転させる
            array = get_average_features(self.X, self.labels, n).values
            extended_features = np.tile(-array, (self.k, 1))
        else:
            # 元の特徴量行列Xから適当にコピーして特徴量行列を作る
            extended_features = self.X[np.random.choice(np.arange(self.X.shape[0]), self.k, replace=False)]

        # 生成した特徴量行列をテスト
        test_features_matrix(extended_features, self.k)
        return extended_features

    def __del__(self):

        pass
    
    
class GeneticAttacker(BaseAttacker):
    """
    エッジの組み合わせを遺伝アルゴリズムで最適化する攻撃
    - 組み合わせの数を削減するため網羅ランダムにする
    - 学習済みのGCNの予測結果とLightGBMの予測精度を比較して精度の低下具合を評価する
    - 最も精度が低下するエッジの組み合わせを探索する
    
    
    Notes
    -----
    遺伝子(gene) : 一つの設計変数
    個体(individual) : 設計変数の1セット
    個体集合(population) : 個体を集めたセット。現世代(population)と次世代(offspring)の2つを用意する必要があります。
    世代(generation) : 現世代と次世代を包括した個体集合の表現。
    適応度(fitness) : 各個体に対する目的関数の値。
    選択(selection) : 現世代から次世代への淘汰のこと。適応度の高いものを優先的に選択します。
    交叉(crossover) : 2個体間の遺伝子の入れ替えのこと。生物が交配によって子孫を残すことをモデル化したもの。
    突然変異(mutation) : 個体の遺伝子をランダムに変化させること。
    """

    def __init__(self, data, population_size, generation, mutate_rate, elite_rate):
        super().__init__(data)
        self.population_size = population_size
        self.generation = generation
        self.mutate_rate = mutate_rate
        self.elite_rate = elite_rate
        
    def _fitness(self, arr, X_all, labels_all, model):
        """適応度（GCNの精度）を得る関数"""
        A_all = stack_adj(arr, self.A)

        data_all = transform(A_all, X_all, labels_all)
        data_all.test_indices = self.data.test_indices
        data_all.test_mask = masking(data_all.test_indices, data_all.num_nodes)

        # 推論
        test_acc, val_loss = test(model, data_all, self.device)

        return test_acc

    def _evaluate(self, population):
        """scoreの低いもの上位20%を持ってくる"""
        print([x[0] for x in population])
        population.sort(key=lambda x:x[0])
        return population[:int(self.elite_rate*len(population))]

    def _get_individual(self):
        """個体として「500×50000のarr」を生成する"""
        arr = np.zeros((self.k, const.TESTSIZE))
        target_indices = np.arange(const.TESTSIZE)
    
        rand_index = np.random.choice(target_indices, len(target_indices), replace=False)

        for v in arr:
            # ランダムにe個選ぶ
            #v[np.random.choice(target_indices, e, replace=False)]=1
            # 網羅的にe個選ぶ
            v[rand_index[:100]] = const.ATTACK_VALUE
            rand_index = rand_index[100:]
        return arr
    
    def _get_population(self, X_all, labels_all, model):
        """個体を生成してfitnessによる適応度を算出する"""
        population = []
        for i in np.arange(self.population_size):
            arr = self._get_individual()
            score = self._fitness(arr, X_all, labels_all, model)
            population.append([score, arr])
        return population

    def _crossover_shift(self, parent, partition):
        """網羅ランダムアタック用の交叉
        - 500*50000の行列をpartitionで左右に分割する
        - ランダムに左右のどちらかを反転させる
            - 上下で反転すると一行あたり100エッジの制約を満たせなくなる
        - 突然変異はpartition=0.8    
        """
        #pivot = int(parent.shape[1]*partition)
        if(np.random.rand()>0.5):
            parent[:, :partition] = np.fliplr(parent[:, :partition]).copy()
        else:
            parent[:, -partition:] = np.fliplr(parent[:, -partition:]).copy()
        return parent

    def _mutate_shift(self, parent, partition=0.8):
        return self._crossover_shift(parent, partition)

    def _mutate_rand(self, parent):
        return _get_individual()
    
    def genetic_attack(self, model, extended_features):
        """遺伝アルゴリズム"""
        print("##### Genetic Attack #####")
        # A_allはそれぞれのループごとに計算する、X_allとlabels_allは固定
        X_all = vstack([self.X, extended_features]).toarray()
        labels_all = np.append(self.labels, np.ones(500)*(-1))
        
        print('Generation: 0')
        # populationの初期化
        population = self._get_population(X_all, labels_all, model)
        elites = self._evaluate(population)
        
        # GPUメモリ対策
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 遺伝的アルゴリズム
        for g in np.arange(1, self.generation):
            print(f'Generation: {g} / {self.generation}')
            print(f'TOP 5/{len(elites)} ELITES INDIVIDUAL: {[x[0] for x in elites[:5]]}')
            # 突然変異、交叉
            pop = elites.copy()
            i = 0
            while len(pop) < self.population_size:
                i_cyclic = int(i % len(elites))
                if np.random.rand() < self.mutate_rate:
                    # 20%の確率で突然変異させる
                    print(f"{i}: Mutation !")
                    #child = self._mutate_shift(elites[i_cyclic][1], 0.8)
                    child = self._get_individual()
                else:
                    # 一様交叉させる
                    print(f"{i}: Cross Over !")
                    partition =np.random.randint(0, int(const.ADJSIZE/4))
                    child = self._crossover_shift(elites[i_cyclic][1], partition)
                pop.append([self._fitness(child, X_all, labels_all, model), child])
                i += 1
            # 評価
            # 上位20%をエリートとする
            elites = self._evaluate(pop)
        print("##### Genetic Attack Result #####")
        print(f'TOP 5/{len(elites)} ELITES INDIVIDUAL: {[x[0] for x in elites[:5]]}')
        return elites
        
        
    def generate_adj(self, elites):
        # k*kの隣接行列を作る（新規ノード同士は接続させない）
        arr_right = np.zeros((self.k, self.k))
        csr_right = csr_matrix(arr_right)

        arr_mid = elites[0][1]
        csr_mid = csr_matrix(arr_mid)

        arr_left = np.zeros((self.k, const.TRAINSIZE))
        csr_left = csr_matrix(arr_left)

        # 連結する
        extended_adj = hstack([csr_left, csr_mid ,csr_right], format='csr')

        test_adjacent_matrix(extended_adj, const.ADJSIZE)

        return extended_adj


    def generate_features(self, kind="random", n=3):
        """kindに応じて攻撃用の特徴量を生成する"""
        
        if kind=="100":
            extended_features = 100*np.ones((self.k, self.e))
        elif kind=="100/-100":
            arr=np.append(np.ones(25000)*self.e, np.ones(25000)*(-self.e))
            np.random.shuffle(arr)
            extended_features = arr.reshape((self.k, self.e))
        elif kind=="zero":
            extended_features = np.zeros((self.k, self.e))
        elif kind=="gauss":
            mu, sigma = 0, 1
            extended_features = np.random.normal(mu, sigma, (self.k, self.e))
        elif kind=="random":
            extended_features = np.random.rand(self.k, self.e)
        elif kind=="target":
            array = get_average_features(self.X, self.labels, n).values
            extended_features = np.tile(array, (self.k, 1))
        elif kind=="target_inverse":
            # 符号を反転させる
            array = get_average_features(self.X, self.labels, n).values
            extended_features = np.tile(-array, (self.k, 1))
        else:
            # 元の特徴量行列Xから適当にコピーして特徴量行列を作る
            extended_features = self.X[np.random.choice(np.arange(self.X.shape[0]), self.k, replace=False)]

        # 生成した特徴量行列をテスト
        test_features_matrix(extended_features, self.k)
        return extended_features


    def __del__(self):
        pass
    


class GradientAttacker(BaseAttacker):
    """
    勾配計算から特徴量行列を改竄する攻撃
    """
    def __init__(self, data):
        super().__init__(data)
        self.victim_model, self.params = load_optimized_model("./tests/pyg_model/",
                                                              data, 
                                                              device,
                                                              fname_model,
                                                              fname_params
                                                             )

    def init_features(self):
        """特徴量行列をランダムに(-1,1)で初期化"""
        extended_X = np.random.randn(self.k, self.e)
        X_all = vstack([self.X, extended_X]).toarray()
        X_all = torch.tensor(X_all, dtype=torch.float, requires_grad=True)
        return X_all
    
    
    def generate_data_attacked(self, A, X, labels):
        data_attacked = transform(A, X, labels)
        # テスト用のインデックス、マスキングを設定
        data_attacked.test_indices = np.arange(const.TRAINSIZE, const.ADJSIZE)
        data_attacked.test_mask = masking(data_attacked.test_indices, data_attacked.num_nodes)
        return data_attacked
    
    def gradient_attack(self, N=10):
        """勾配計算から特徴量行列の値を改竄する"""
        print("##### Genetic Attack #####")
        for turn in np.arange(N):
            print(f"Turn {turn} start...")
            # Lossを計算する前にrequired_gradフラグをTrueに設定する
            data_attacked.x.requires_grad_(True)
            # AccとLossを計算する
            test_acc, test_loss = test(self.victim_modle,
                                       data_attacked,
                                       device    
                                      )
            print(f'Test Loss: {test_loss:.4f}, Test: {test_acc:.4f}')
            
            # 微分計算 Lossを最小化する方向の特徴量行列の変化量をみる
            grad = torch.autograd.grad(test_loss, data_attacked.x, retain)
            
            # 求めた勾配の逆向き（Lossを最大化する方向）を考えて、特徴量を改竄していく
            for inx in np.arange(const.MAX_ADD_NODE):
                line = const.TRAINSIZE + inx
                if(inx%100==0):
                    print(f"Gradient Attacking... (turn, inx) = ({turn}, {inx})")

                for dim in np.arange(const.FEATURE_DIM):
                    if grad[line][dim] > 0:
                        # 勾配が正なので同じ方向に摂動を加える
                        data_attacked.x[line, dim] = data_attacked.x[line, dim]+1.0
                        if data_attacked.x[line, dim] > 2.0:
                            # 無限に大きくなってしまうので上限2.0とする
                            data_attacked.x[line, dim] = 2.0
                    elif grad[line][dim] < 0:
                        data_attacked.x[line, dim] = data_attacked.x[line, dim]-1.0
                        if data_attacked.x[line, dim] < -2.0:
                            data_attacked.x[line, dim] = -2.0
            data_attacked = self.generate_data_attacked(A, data_attacked.x, labels_all)
        print("done")
                            
   
    def __del__(self):
        pass
    


##### 

    
def stack_adj(extended_arr, A):
    """読み込んだ隣接行列(593486*593486)に攻撃用の行列(500*50000)を組み合わせる関数"""
    arr_lower = hstack([csr_matrix(np.zeros((const.MAX_ADD_NODE, const.TRAINSIZE))),
                       csr_matrix(extended_arr),
                       csr_matrix(np.zeros((const.MAX_ADD_NODE, const.MAX_ADD_NODE)))],
                       format="csr")
    arr_upper = hstack([A, arr_lower.T[:-const.MAX_ADD_NODE,:]], format="csr")
    A_all = vstack([arr_upper, arr_lower])
    return A_all

def transform(A, X, labels):
    # PytorchのTensor型に変換する
    edge_index, edge_attr = from_scipy_sparse_matrix(A)
    print(type(X), type(labels))
    x = torch.tensor(X, dtype=torch.float)
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


##### test用の関数 #####



def test_adjacent_matrix(arr, adjsize):
    """提出する隣接行列の対称性とエッジ数をテスト"""
    # 新規ノード部分の対称性チェック
    if(arr[:, adjsize:]-arr[:, adjsize:].T).nnz==0:
        print("[OK] : adj_matrix is symmetric.")
    else:
        raise ValueError("[NG] : adj_matrix is not symmetric. Diff {}".format((arr-arr.T).nnz))

    # １ノードあたりのエッジ数が100以下かどうかチェック
    e_max, e_min = arr.getnnz(axis=1).max(), arr.getnnz(axis=1).min()
    e_ave, e_var = arr.getnnz(axis=1).mean(), arr.getnnz(axis=1).var()
    if arr.getnnz(axis=1).max()<=100:
        print("[OK] : (max, min, ave, var) = ({}, {}, {}, {})".format(e_max, e_min, e_ave, e_var))
    else:
        raise ValueError("[NG] : (max, min, ave, var) = ({}, {}, {}, {})".format(e_max, e_min, e_ave, e_var))

    print("😄 Adjacent_matrix is OK !")


def test_features_matrix(arr, k):
    """提出する特徴量行列のサイズをテスト"""
    if(arr.shape!=(k,100)):
        raise ValueError("[NG] : features_matrix shape is {}".format(arr.shape))
    else:
        print("[OK] : features_matrix shape is {}".format(arr.shape))
    
    if((arr.max()<=100) and(arr.min()>=-100)):
        print("[OK] : features_matrix (max, min, ave, var) = ({}, {}, {}, {})".format(arr.max(), 
                                                                                      arr.min(), 
                                                                                      arr.mean(), 
                                                                                      arr.var()))
    else:
        raise ValueError("[NG] : features_matrix value is out of range... (max, min, ave, var) = ({}, {}, {}, {})".format(arr.max(), arr.min(), arr.mean(), arr.var()))
    
    print("😄 Features_matrix is OK !")