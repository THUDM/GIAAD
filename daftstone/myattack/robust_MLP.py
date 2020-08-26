import tensorflow as tf
import random
import math
import numpy as np
from myattack import utils
import scipy.sparse as sp
from tensorflow.contrib import slim
import os
import pickle
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, desc=None: x


class Robust_GCN:
    """
            Base class for attacks on GNNs.
        """

    def __init__(self, adjacency_matrix_tuple, attribute_matrix, labels_onehot_T, gpu_id=None,
                 weight_decay=5e-4, learning_rate=0.005, dropout=0.2):
        self.attribute_matrix = attribute_matrix
        self.adjacency_matrix_tuple = adjacency_matrix_tuple
        self.K = labels_onehot_T.shape[1]
        self.N, self.D = attribute_matrix.shape
        self.graph = tf.Graph()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.cood = self.adjacency_matrix_tuple[0]
        self.value = self.adjacency_matrix_tuple[1]

        with self.graph.as_default():
            self.training = tf.placeholder_with_default(False, shape=())

            self.idx = tf.placeholder(tf.int32, shape=[None])
            self.val_idx = tf.placeholder(tf.int32, shape=[None])
            self.true_label_onehot = labels_onehot_T

            self.adj_value = tf.placeholder(tf.float32)
            self.sparse_slice = tf.placeholder(tf.int64)
            self.adj_norm = tf.SparseTensor(self.sparse_slice,
                                            self.adj_value, [self.N, self.N])

            self.attributes1 = tf.placeholder(tf.float32)
            self.attributes = (self.attributes1 - tf.reduce_min(self.attributes1, axis=1,keepdims=True)) / (
                        tf.reduce_max(self.attributes1, axis=1,keepdims=True) - tf.reduce_min(self.attributes1, axis=1,keepdims=True) + 1e-8)
            # mean, var = tf.nn.moments(self.attributes1, axes=[1], keep_dims=True)
            # self.attributes = (self.attributes1 - mean) / (tf.sqrt(var) + 1e-8)

            self.attributes_dropout = tf.nn.dropout(self.attributes, rate=dropout)

            self.w_init = slim.xavier_initializer
            if gpu_id is None:
                config = tf.ConfigProto(
                    device_count={'GPU': 0}
                )
            else:
                gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id))
                config = tf.ConfigProto(gpu_options=gpu_options)
            session = tf.Session()
            self.session = session

            self.build_v0(drop=dropout)
            self.logits_all = self.logits

    def build_v0(self, with_relu=True, hidden_sizes=[32, 32, 32, 32], drop=0.1):
        with self.graph.as_default():
            self.weights = []
            self.biases = []

            previous_size = self.D
            for ix, layer_size in enumerate(hidden_sizes):
                weight = tf.Variable(
                    tf.truncated_normal(shape=[previous_size, layer_size], mean=0.0, stddev=0.01),
                    name='w_%d' % (ix + 1), dtype=tf.float32)
                bias = tf.Variable(
                    tf.zeros(shape=(layer_size)),
                    name='b_%d' % (ix + 1), dtype=tf.float32)
                self.weights.append(weight)
                self.biases.append(bias)
                previous_size = layer_size

            weight = tf.Variable(
                tf.truncated_normal(shape=[previous_size, self.K], mean=0.0, stddev=0.01),
                name='w_%d' % (len(hidden_sizes) + 1), dtype=tf.float32)
            bias = tf.Variable(
                tf.zeros(shape=[self.K]),
                name='b_%d' % (len(hidden_sizes) + 1), dtype=tf.float32)
            self.weights.append(weight)
            self.biases.append(bias)

            hidden = tf.cond(self.training,
                             lambda: self.attributes_dropout,
                             lambda: self.attributes) if drop > 0. else self.attributes

            layers = []
            for ix in range(len(hidden_sizes)):
                w = self.weights[ix]
                b = self.biases[ix]

                hidden = tf.matmul(hidden, w) + b

                # if(ix>=2):
                #     mean, var = tf.nn.moments(hidden, axes=[0], keep_dims=True)
                #     hidden = (hidden - mean) / (tf.sqrt(var) + 1e-8)

                if with_relu:
                    hidden = tf.nn.leaky_relu(hidden)

                hidden_dropout = tf.nn.dropout(hidden, rate=drop)
                hidden = tf.cond(self.training,
                                 lambda: hidden_dropout,
                                 lambda: hidden) if drop > 0. else hidden

            self.logits = tf.matmul(hidden, self.weights[-1]) + self.biases[-1]

            self.logits_gather = tf.gather(self.logits, self.idx)
            labels_gather = tf.gather(self.true_label_onehot, self.idx)

            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather, logits=self.logits_gather)
            self.loss += self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.weights])
            self.loss += self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.biases])

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, var_list=[*self.weights, *self.biases])

            self.logits_gather1_v1 = tf.gather(self.logits, self.val_idx)
            labels_gather1 = tf.gather(self.true_label_onehot, self.val_idx)
            acc_count = tf.equal(tf.argmax(tf.nn.softmax(self.logits_gather1_v1), axis=-1),
                                 tf.argmax(labels_gather1, axis=-1))
            acc_count = tf.cast(acc_count, tf.float32)
            self.acc = tf.reduce_mean(acc_count)

    def build_v1(self, with_relu=True, hidden_sizes=[128, 128, 64, 32], drop=0.1):
        with self.graph.as_default():
            self.weights1 = []
            self.biases1 = []

            previous_size = self.D
            for ix, layer_size in enumerate(hidden_sizes):
                weight = tf.Variable(
                    tf.truncated_normal(shape=[previous_size, layer_size], mean=0.0, stddev=0.01),
                    name='w1_%d' % (ix + 1), dtype=tf.float32)
                bias = tf.Variable(
                    tf.zeros(shape=(layer_size)),
                    name='b1_%d' % (ix + 1), dtype=tf.float32)
                self.weights1.append(weight)
                self.biases1.append(bias)
                previous_size = layer_size

            weight = tf.Variable(
                tf.truncated_normal(shape=[previous_size, self.K], mean=0.0, stddev=0.01),
                name='w1_%d' % (len(hidden_sizes) + 1), dtype=tf.float32)
            bias = tf.Variable(
                tf.zeros(shape=[self.K]),
                name='b1_%d' % (len(hidden_sizes) + 1), dtype=tf.float32)
            self.weights1.append(weight)
            self.biases1.append(bias)

            hidden = tf.cond(self.training,
                             lambda: self.attributes_dropout,
                             lambda: self.attributes) if drop > 0. else self.attributes

            layers = []
            for ix in range(len(hidden_sizes)):
                w = self.weights1[ix]
                b = self.biases1[ix]

                hidden1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
                # hidden1 = tf.concat([hidden, hidden1], axis=-1)
                hidden = tf.matmul(hidden1, w) + b

                if (ix >= 2):
                    mean, var = tf.nn.moments(hidden, axes=[0], keep_dims=True)
                    hidden = (hidden - mean) / (tf.sqrt(var) + 1e-8)
                if with_relu:
                    hidden = tf.nn.leaky_relu(hidden)

                hidden_dropout = tf.nn.dropout(hidden, rate=drop)
                hidden = tf.cond(self.training,
                                 lambda: hidden_dropout,
                                 lambda: hidden) if drop > 0. else hidden
                layers.append(hidden)

            hidden = tf.stack(layers, axis=0)
            hidden = tf.reduce_max(hidden, axis=0)

            hidden1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
            # hidden1 = tf.concat([hidden, hidden1], axis=-1)
            self.logits1 = tf.matmul(hidden1, self.weights1[-1]) + self.biases1[-1]

            self.logits_gather1 = tf.gather(self.logits1, self.idx)
            labels_gather1 = tf.gather(self.true_label_onehot, self.idx)

            self.loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather1, logits=self.logits_gather1)
            self.loss1 += self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.weights1])
            self.loss1 += self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.biases1])

            self.optimizer1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op1 = self.optimizer1.minimize(self.loss1, var_list=[*self.weights1, *self.biases1])

            self.logits_gather1_v2 = tf.gather(self.logits1, self.val_idx)
            labels_gather2 = tf.gather(self.true_label_onehot, self.val_idx)
            acc_count = tf.equal(tf.argmax(tf.nn.softmax(self.logits_gather1_v2), axis=-1),
                                 tf.argmax(labels_gather2, axis=-1))
            acc_count = tf.cast(acc_count, tf.float32)
            self.acc1 = tf.reduce_mean(acc_count)

    def train(self, idx_train, idx_val, n_iters=200, initialize=True, display=True):
        with self.graph.as_default():
            if initialize:
                self.session.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()

            _iter = range(n_iters)
            if display:
                _iter = tqdm(_iter, desc="Training")
            self.attribute_matrix[659574:] = 0
            for _it in _iter:
                ttt = self.session.run(
                    [self.train_op, self.acc],
                    feed_dict={self.idx: idx_train, self.training: True, self.val_idx: idx_val,
                               self.adj_value: self.value, self.sparse_slice: self.cood,
                               self.attributes1: self.attribute_matrix})
                    # print(ttt[1])
                _iter.set_description("acc: %f" % (ttt[1]))
            self.saver.save(self.session, "model/gcn.ckpt")

    def get_logits(self, adj_tuple, feature):
        # adj_norm = utils.preprocess_graph(adj).astype("float32")
        # cood, value, shape = utils.sparse_to_tuple1(adj_norm)
        with self.graph.as_default():
            # self.session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(self.session, "model/gcn.ckpt")
            tt = self.session.run([self.logits_all],
                                  feed_dict={self.training: False, self.adj_value: adj_tuple[1],
                                             self.sparse_slice: adj_tuple[0],
                                             self.attributes1: feature})
            return tt[0]
