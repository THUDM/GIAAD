"""
KDDCUP2020 MLTrack2
https://www.biendata.xyz/competition/kddcup_2020/

Author: NTT DOCOMO LABS
License: MIT
"""

import sys
import os
import pickle
import random
import zipfile
import datetime

import numpy as np
import torch

from d_attack.models import GCN, SAGE, Cheb, GIN, GCN_dev


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    # Tensorflow
    #tf.random.set_seed(seed)


def load_fdict(flist):
    """引数のリストから指定したファイルを読み込む関数"""
    fdict = {}
    for s in flist:
        if s.endswith("adj.pkl"):
            print(f"add {s} in fdict...")
            fdict["adj"] = s
        elif s.endswith("feature.npy"):
            print(f"add {s} in fdict...")
            fdict["feature"] = s
        elif s.endswith("output.csv"):
            print(f"add {s} in fdict...")
            fdict["output"] = s
        else:
            print(f"pass {s}...")
            #raise NameError("Failed loading filename...")
    return fdict


def load_optimized_model(path, data, device, fname_model, fname_params, kind):
    """学習済みのモデルを読み込んでPyGモデルとパラメータを返す"""
    with open(path+fname_params, "rb") as f:
        params = pickle.load(f)
    print(params)

    model = eval(kind)(data.num_node_features,
                       data.num_class,
                       params['hidden'],
                       params['dropout_rate']).to(device)

    model.load_state_dict(torch.load(path+fname_model, map_location=device))

    return model, params

def overwrite_empty_features(result, features):
    empty_features_inx = np.where(np.all(features==0, axis=1))
    result[empty_features_inx] = 1
    print(f"Detect ALL 0 features !", empty_features_inx)
    return result