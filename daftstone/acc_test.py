from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import pickle
import numpy as np
import math
import scipy.sparse as sp
from myattack import utils
import tensorflow as tf
from myattack import GCN

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('adj', 'adj.pkl', '')
flags.DEFINE_string('feature', 'feature.npy', '')
flags.DEFINE_string('gpu', '0', 'gpu id')
flags.DEFINE_string('id', '0', 'gpu id')
flags.DEFINE_string('pred', '1', 'gpu id')
flags.DEFINE_bool("train", False, "pass")
flags.DEFINE_bool("relu", False, "pass")
flags.DEFINE_bool("norm1", False, "pass")
flags.DEFINE_bool("norm2", False, "pass")
flags.DEFINE_bool("add", False, "pass")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
hidden_sizes = [100]
train_iters = 400
improved = 2
dropout = 0.2
# relu=True
relu = FLAGS.relu
train = FLAGS.train
gpu_id=FLAGS.gpu


# train=True
def norm(xxx, train=False):
    x = xxx.copy()
    # x=np.arctan(x)*2/np.pi
    # x=(x-np.mean(x))/np.std(x)
    # x=x/np.max(np.abs(x))
    if (train == False):
        xx = x[593486:].copy()
        # a = np.log10(np.abs(xx) + 1)
        # a[xx < 0] = 0
        # b = -np.log10(np.abs(xx) + 1)
        # b[xx >= 0] = 0
        # x[593486:] = a + b
        a = np.min(x[:593486], axis=0, keepdims=True)
        b = np.max(x[:593486], axis=0, keepdims=True)
    if (FLAGS.norm1 == True):
        x[:593486] = (x[:593486] - np.min(x[:593486],axis=1,keepdims=True)) / (np.max(x[:593486],axis=1,keepdims=True)-np.min(x[:593486],axis=1,keepdims=True))*2-1
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    return x


length=659574
_A_obs, _X_obs, _z_obs = utils.load_pkl()
_z_obs[_z_obs==18]=0
_z_obs[_z_obs==19]=2
_z_obs_true=np.concatenate([_z_obs,np.zeros(50500,dtype=np.int)])
_A_obs = _A_obs.astype("float32")
_A_obs.eliminate_zeros()
_X_obs = _X_obs.astype("float32")

np.random.seed(0)
idx=np.arange(0,length-50000)
np.random.shuffle(idx)
split_train=idx[:600000]
split_val=idx[600000:length-50000]
split_test=np.arange(length-50000,length)

add_X_obs=np.zeros((500,100),dtype=np.float)
_X_obs=np.concatenate([_X_obs,add_X_obs],axis=0)
add_A_obs, _ = utils.generate_node(length,begin=length-50000)
_A_obs_origin=_A_obs.copy()
_A_obs = sp.vstack([_A_obs, add_A_obs])
A_temp = sp.vstack([_A_obs.T[:, length:], sp.csr_matrix(np.zeros((500, 500)))])
_A_obs = sp.hstack([_A_obs, A_temp])
_A_obs = _A_obs.tocsr()

# create clean adj
add_A_obs1=sp.csr_matrix(np.zeros((500,length)))
_A_obs1 = sp.vstack([_A_obs_origin, add_A_obs1])
A_temp = sp.vstack([_A_obs1.T[:, length:], sp.csr_matrix(np.zeros((500, 500)))])
_A_obs1 = sp.hstack([_A_obs1, A_temp])
_A_obs_clean = _A_obs1.tocsr()


print("pure", np.max(_A_obs_clean[659574:].sum(1)))
adj_norm_clean = utils.preprocess_graph(_A_obs_clean).astype("float32")
# z=np.load("label.npy")
_Z_obs=np.eye(18)[_z_obs_true]
surrogate = GCN.GCN(_A_obs_clean, norm(_X_obs, train=True), _Z_obs, hidden_sizes=hidden_sizes, gpu_id=0,
                    dropout=dropout, improved=improved)
surrogate.build(with_relu=relu)
if (train):
    surrogate.train(split_train, n_iters=train_iters, model="../temp/gcn2_%s.ckpt" % FLAGS.id)
surrogate.saver.restore(surrogate.session, "../temp/gcn2_%s.ckpt" % FLAGS.id)

pred = surrogate.logits.eval(session=surrogate.session)
pred = pred.argmax(1)
_z_obs_true[length-50000:length] = pred[length-50000:length]
print("clean")
print("accuracy on validation data ",
      (pred == _z_obs_true)[split_val].mean())
print("accuracy on test data", (pred == _z_obs_true)[split_test].mean())

print("test")
with open(FLAGS.adj, "rb+") as f:
    adj = pickle.load(f)
    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
adj_norm = utils.preprocess_graph(adj).astype("float32")
feature = np.load(FLAGS.feature)
_X_obs_noise = _X_obs.copy()
_X_obs_noise[659574:] = feature
pred = surrogate.session.run(surrogate.logits_placeholder,
                             feed_dict={surrogate.slice_placeholder: np.array(adj_norm.nonzero()).T,
                                        surrogate.value_placeholder: adj_norm[adj_norm.nonzero()].A1,
                                        surrogate.attributes_placeholder: norm(_X_obs_noise),
                                        surrogate.training: False})
pred = pred.argmax(1)
print("accuracy on perturbation validation data ",
      (pred == _z_obs_true)[split_val].mean())
print("accuracy on test data", (pred == _z_obs_true)[split_test].mean())

