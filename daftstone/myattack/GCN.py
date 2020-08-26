import tensorflow as tf
import random
import numpy as np
from myattack import utils
import scipy.sparse as sp
from tensorflow.contrib import slim

flags = tf.flags
FLAGS = flags.FLAGS
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, desc=None: x


class GCN:
    """
            Base class for attacks on GNNs.
        """

    def __init__(self, adjacency_matrix, attribute_matrix, labels_onehot, hidden_sizes, gpu_id=None,
                 weight_decay=5e-4, learning_rate=0.005, dropout=0.5, improved=1):
        """
        Parameters
        ----------
        adjacency_matrix: sp.spmatrix [N,N]
                Unweighted, symmetric adjacency matrix where N is the number of nodes. Should be a scipy.sparse matrix.

        attribute_matrix: sp.spmatrix or np.array [N,D]
            Attribute matrix where D is the number of attributes per node. Can be sparse or dense.

        labels_onehot: np.array [N,K]
            One-hot matrix of class labels, where N is the number of nodes. Labels of the unlabeled nodes should come
            from self-training using only the labels of the labeled nodes.

        hidden_sizes: list of ints
            List that defines the number of hidden units per hidden layer. Input and output layers not included.

        gpu_id: int or None
            GPU to use. None means CPU-only

        weight_decay: float, default 5e-4
            L2 regularization for the first layer only (matching the original implementation of GCN)

        learning_rate: float, default 0.01
            The learning rate used for training.

        dropout: float, default 0.5
            Dropout used for training.

        """
        if not sp.issparse(adjacency_matrix):
            raise ValueError("Adjacency matrix should be a sparse matrix.")

        self.imporved = improved
        self.N, self.D = attribute_matrix.shape
        self.K = labels_onehot.shape[1]
        self.hidden_sizes = hidden_sizes
        self.graph = tf.Graph()

        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay

        with self.graph.as_default():
            self.training = tf.placeholder_with_default(False, shape=())

            self.idx = tf.placeholder(tf.int32, shape=[None])
            self.labels_onehot = labels_onehot

            adj_norm = utils.preprocess_graph(adjacency_matrix).astype("float32")
            self.adj_norm = tf.SparseTensor(np.array(adj_norm.nonzero()).T,
                                            adj_norm[adj_norm.nonzero()].A1, [self.N, self.N])

            self.slice_placeholder = tf.placeholder(tf.int64)
            self.value_placeholder = tf.placeholder(tf.float32)
            self.adj_norm_placeholder = tf.SparseTensor(self.slice_placeholder, self.value_placeholder,
                                                        [self.N, self.N])
            self.attributes_placeholder = tf.placeholder(tf.float32)

            self.sparse_attributes = sp.issparse(attribute_matrix)

            if self.sparse_attributes:
                self.attributes = tf.SparseTensor(np.array(attribute_matrix.nonzero()).T,
                                                  attribute_matrix[attribute_matrix.nonzero()].A1, [self.N, self.D])
                self.attributes_dropout = sparse_dropout(self.attributes, 1 - self.dropout,
                                                         (int(self.attributes.values.get_shape()[0]),))
            else:
                self.attributes = tf.Variable(attribute_matrix, dtype=tf.float32)
                self.attributes_dropout = tf.nn.dropout(self.attributes, rate=dropout)

            self.attrs_comp = tf.cond(self.training,
                                      lambda: self.attributes_dropout,
                                      lambda: self.attributes) if self.dropout > 0. else self.attributes

            w_init = slim.xavier_initializer
            self.weights = []
            self.biases = []

            previous_size = self.D
            for ix, layer_size in enumerate(self.hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}", shape=[previous_size * self.imporved, layer_size],
                                         dtype=tf.float32,
                                         initializer=w_init())
                bias = tf.get_variable(f"b_{ix + 1}", shape=[layer_size], dtype=tf.float32,
                                       initializer=w_init())
                self.weights.append(weight)
                self.biases.append(bias)
                previous_size = layer_size

            weight_final = tf.get_variable(f"W_{len(hidden_sizes) + 1}", shape=[previous_size * self.imporved, self.K],
                                           dtype=tf.float32,
                                           initializer=w_init())
            bias_final = tf.get_variable(f"b_{len(hidden_sizes) + 1}", shape=[self.K], dtype=tf.float32,
                                         initializer=w_init())

            self.weights.append(weight_final)
            self.biases.append(bias_final)

            if gpu_id is None:
                config = tf.ConfigProto(
                    device_count={'GPU': 0}
                )
            else:
                gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)

            session = tf.Session(config=config)
            self.session = session

            self.logits = None
            self.logits_gather = None
            self.loss = None
            self.optimizer = None
            self.train_op = None
            self.initializer = None

    def build(self, with_relu=True):
        with self.graph.as_default():
            hidden = tf.cond(self.training,
                             lambda: self.attributes_dropout,
                             lambda: self.attributes) if self.dropout > 0. else self.attributes
            hidden_place = self.attributes_placeholder

            hidden_temp = 0
            hidden_place_temp = 0
            layers = []
            layers_holder = []
            for ix in range(len(self.hidden_sizes)):
                w = self.weights[ix]
                b = self.biases[ix]

                hidden1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
                if (FLAGS.add):
                    hidden1 += hidden
                if (self.imporved == 2):
                    hidden1 = tf.concat([hidden1, hidden], axis=-1)
                hidden = hidden1 @ w + b
                hidden2 = tf.sparse_tensor_dense_matmul(self.adj_norm_placeholder, hidden_place)
                if (FLAGS.add):
                    hidden2 += hidden_place
                if (self.imporved == 2):
                    # hidden2 = (hidden2 + hidden_place)
                    hidden2 = tf.concat([hidden2, hidden_place], axis=-1)
                hidden_place = hidden2 @ w + b

                if (FLAGS.norm2):
                    mean, variance = tf.nn.moments(hidden, axes=[1], keep_dims=True)
                    hidden = (hidden - mean) / tf.sqrt(variance)
                    mean, variance = tf.nn.moments(hidden_place, axes=[1], keep_dims=True)
                    hidden_place = (hidden_place - mean) / tf.sqrt(variance)

                # hidden+=hidden_temp
                # hidden_place+=hidden_place_temp

                if with_relu:
                    hidden = tf.nn.relu(hidden)
                    hidden_place = tf.nn.relu(hidden_place)

                layers.append(hidden)
                layers_holder.append(hidden_place)

                hidden_dropout = tf.nn.dropout(hidden, rate=self.dropout)
                hidden = tf.cond(self.training,
                                 lambda: hidden_dropout,
                                 lambda: hidden) if self.dropout > 0. else hidden

            hidden = tf.stack(layers, axis=0)
            hidden = tf.reduce_max(hidden, axis=0)
            hidden_place = tf.stack(layers_holder, axis=0)
            hidden_place = tf.reduce_max(hidden_place, axis=0)

            hidden1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
            if (FLAGS.add):
                hidden1+=hidden
            if (self.imporved == 2):
                hidden1 = tf.concat([hidden1, hidden], axis=-1)
            self.logits = hidden1 @ self.weights[-1] + self.biases[-1]
            logits2 = tf.sparse_tensor_dense_matmul(self.adj_norm_placeholder, hidden_place)
            if (FLAGS.add):
                logits2 += hidden_place
            if (self.imporved == 2):
                logits2 = tf.concat([logits2, hidden_place], axis=-1)
            self.logits_placeholder = logits2 @ self.weights[-1] + self.biases[-1]

            self.logits_gather = tf.gather(self.logits, self.idx)
            labels_gather = tf.gather(self.labels_onehot, self.idx)

            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather, logits=self.logits_gather)
            self.loss += self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in [self.weights[0], self.biases[0]]])

            logits_val_gather = tf.gather(self.logits, np.arange(559574, 609574))
            labels_val_gather = tf.gather(self.labels_onehot, np.arange(559574, 609574))
            acc_count = tf.equal(tf.argmax(tf.nn.softmax(logits_val_gather), axis=-1),
                                 tf.argmax(labels_val_gather, axis=-1))
            acc_count = tf.cast(acc_count, tf.float32)
            self.acc = tf.reduce_mean(acc_count)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, var_list=[*self.weights, *self.biases])
            self.initializer = tf.local_variables_initializer()
            self.saver = tf.train.Saver()
            self.session.run(tf.global_variables_initializer())

    def train(self, idx_train, n_iters=200, initialize=True, display=True, model='model/gcn1.ckpt'):
        with self.graph.as_default():
            if initialize:
                self.session.run(tf.global_variables_initializer())

            _iter = range(n_iters)
            if display:
                _iter = tqdm(_iter, desc="Training")

            for _it in _iter:
                tt = self.session.run([self.train_op, self.acc], feed_dict={self.idx: idx_train, self.training: True})
                # print(tt[1])
                _iter.set_description("acc: %f" % (tt[1]))
            self.saver.save(self.session, model)


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def layernorm(x, offset, scale):
    mean, variance = tf.nn.moments(x, axes=[1], keep_dims=True)
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-9)