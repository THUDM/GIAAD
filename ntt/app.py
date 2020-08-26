"""
KDDCUP2020 MLTrack2
https://www.biendata.xyz/competition/kddcup_2020/

Author: NTT DOCOMO LABS
License: MIT
"""

import sys

sys.path.append("ntt/")
import pickle
import time

import numpy as np
import scipy

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.nn import Linear

from d_attack.utils import fix_seed, load_fdict, load_optimized_model
from d_attack.dataset import load_pyg_data, preprocess_labels
from d_attack.models import GCN, SAGE, Cheb, GIN, GCN_dev, predict

SEED = 42

def nmain(adj,features):
    start = time.time()

    # flistの0番目はapp.pyであることに注意
    #fdict = load_pyg_data(adj,features)
    #print("fdict", fdict)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # データ読み込み
    data = load_pyg_data(adj,features)
    #print(device, data)

    elapsed_time = time.time() - start
    #print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # モデル読み込み
    modelname = "GIN"
    prefix = f"20200721_{modelname}"
    model, params = load_optimized_model("ntt/ML_Model/",
                                         data,
                                         device,
                                         f"{prefix}_model.pth",
                                         f"{prefix}_params.pkl",
                                         modelname
                                        )
    #print(modelname, params)
    
    elapsed_time = time.time() - start
    #print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # 推論処理
    out, pred = predict(model, data, device)
    elapsed_time = time.time() - start
    #print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    

    #print(np.unique(pred.cpu().numpy(), return_counts=True))
    result = preprocess_labels(pred.cpu().numpy(), reverse=True)
    # 特徴量が全て0のノードはラベル1
    #print(np.unique(result, return_counts=True))
    result[tuple(data.empty_features_inx)] = 1
   # print(np.unique(result, return_counts=True))
    return result
    '''
    # 結果格納    
    try:
        np.savetxt(fdict["output"], result, fmt='%d')
    except OSError as e:
        print(e)
    else:
        print("Done!")
        #raise ValueError("INTENTIONAL ERROR!")
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return
    '''


