from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from myattack import utils
from myattack.robust_GCN import Robust_GCN
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('gpu', '0', 'gpu id')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
gpu_id = 0

length = 659574
_A_obs, _X_obs, _z_obs = utils.load_pkl()
_z_obs[_z_obs == 18] = 0
_z_obs[_z_obs == 19] = 2
_z_obs_true = np.load("label.npy")
# _z_obs_true=np.concatenate([_z_obs,np.zeros(50500,dtype=np.int)])
# np.save("label.npy",_z_obs_true)
_A_obs = _A_obs.astype("float32")
_A_obs.eliminate_zeros()
_X_obs = _X_obs.astype("float32")

np.random.seed(0)
idx = np.arange(0, length - 50000)
np.random.shuffle(idx)
split_train = idx[:500000]
split_val = idx[500000:length - 50000]
split_test = np.arange(length - 50000, length)

add_X_obs = np.zeros((500, 100), dtype=np.float)
_X_obs = np.concatenate([_X_obs, add_X_obs], axis=0)
add_A_obs, _ = utils.generate_node(length, begin=length - 50000)
_A_obs_origin = _A_obs.copy()
_A_obs = sp.vstack([_A_obs, add_A_obs])
A_temp = sp.vstack([_A_obs.T[:, length:], sp.csr_matrix(np.zeros((500, 500)))])
_A_obs = sp.hstack([_A_obs, A_temp])
_A_obs = _A_obs.tocsr()
assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"

# create clean adj
add_A_obs1 = sp.csr_matrix(np.zeros((500, length)))
_A_obs1 = sp.vstack([_A_obs_origin, add_A_obs1])
A_temp = sp.vstack([_A_obs1.T[:, length:], sp.csr_matrix(np.zeros((500, 500)))])
_A_obs1 = sp.hstack([_A_obs1, A_temp])
_A_obs_clean = _A_obs1.tocsr()

# x=np.abs(_X_obs)
# m1 = np.max(x, axis=1)
# idx1 = np.argmax(x, axis=1)
# for i in range(idx1.shape[0]):
#     x[i, idx1[i]] = 0
# idx1 = np.argmax(x, axis=1)
# for i in range(idx1.shape[0]):
#     x[i, idx1[i]] = 0
# m2 = np.max(x, axis=1)
# idx = m1 - m2 <= 0.005
# idx2 = np.where(m1 == 0)[0]
# idx[idx2]=False
# if (len(idx) != 0):
#     _X_obs[idx,] = 0
# flag = np.zeros((660074), dtype=np.int)
# if (len(idx) != 0):
#     flag[idx] = 1
#
# _A_obs_clean = _A_obs_clean.tocoo()
# row = _A_obs_clean.row
# col = _A_obs_clean.col
# data = _A_obs_clean.data
# data[flag[row] == 1] = 0
# data[flag[col] == 1] = 0
# _A_obs_clean = sp.coo_matrix((data, (row, col)), shape=(660074, 660074))

_Z_obs_false = np.eye(18)[_z_obs_true]
_Z_obs_true = np.eye(18)[_z_obs_true]

train_iters = 800
dropout = 0.1
learning_rate = 0.01

import pickle

# with open("test/adj.pkl", "rb+") as f:
#     _A_obs_clean = pickle.load(f)
# _X_obs = np.load("test/feature.npy")
_A_obs_clean_norm = utils.preprocess_graph(_A_obs_clean)
_A_obs_clean_tuple = utils.sparse_to_tuple1(_A_obs_clean_norm)
_X_obs = -_X_obs
_X_obs[_X_obs >= 1.8] = 0
_X_obs[_X_obs <= -1.8] = 0
myattack = Robust_GCN(_A_obs_clean_tuple, _X_obs, _Z_obs_true, gpu_id=gpu_id, learning_rate=learning_rate,
                      dropout=dropout)
myattack.train(split_train, split_val, n_iters=train_iters)

preds = myattack.get_logits(_A_obs_clean_tuple, _X_obs)
# preds = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
pred = preds.argmax(1)
_z_obs_true[length - 50000:length] = pred[length - 50000:length]
_Z_obs_true = np.eye(18)[_z_obs_true]
myattack.true_label_onehot = _Z_obs_true
myattack.false_label_onehot = _Z_obs_false
print(pred.shape, _z_obs_true.shape, split_val.shape)
print("accuracy on validation data ",
      (pred == _z_obs_true)[split_val].mean())
print("accuracy on test data ",
      (pred == _z_obs_true)[split_test].mean())
np.save("label.npy",_z_obs_true)
