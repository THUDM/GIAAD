import tensorflow as tf
import random
import math
import numpy as np
from myattack import utils
import scipy.sparse as sp
from tensorflow.contrib import slim
import os
import pickle

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, desc=None: x


class My_Attack:
    """
            Base class for attacks on GNNs.
        """

    def __init__(self, adjacency_matrix, attribute_matrix, labels_onehot_T, labels_onehot_F, gpu_id=None,
                 weight_decay=5e-4, learning_rate=0.005, dropout=0.2, origin_length=593486):
        self.attribute_matrix = attribute_matrix
        self.adjacency_matrix = adjacency_matrix
        self.K = labels_onehot_T.shape[1]
        self.N, self.D = attribute_matrix.shape
        self.graph = tf.Graph()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.adj_norm_sp = utils.preprocess_graph(self.adjacency_matrix).astype("float32")
        self.cood, self.value1, self.value2, self.shape = utils.sparse_to_tuple(self.adj_norm_sp,
                                                                                np.array([609574, 609576]))

        with self.graph.as_default():
            self.training = tf.placeholder_with_default(False, shape=())

            self.idx = tf.placeholder(tf.int32, shape=[None])
            self.val_idx = tf.placeholder(tf.int32, shape=[None])
            self.false_label_onehot = labels_onehot_F
            self.true_label_onehot = labels_onehot_T
            self.idx_attack = tf.placeholder(dtype=tf.int32, shape=[None, ], name="Attack_Idx")

            self.t = tf.Variable(tf.constant(0, dtype=tf.float32))
            self.eps = tf.constant(1e-8)
            self.attr_m = tf.Variable(tf.zeros((500, 100), dtype=tf.float32))
            self.attr_v = tf.Variable(tf.zeros((500, 100), dtype=tf.float32))

            self.adj_value_origin = tf.placeholder(tf.float32)
            self.adj_value_add = tf.placeholder(tf.float32)
            self.sparse_slice = tf.placeholder(tf.int64)
            self.adj_value = tf.concat([self.adj_value_add, self.adj_value_origin], axis=0)
            self.adj_norm = tf.SparseTensor(self.sparse_slice,
                                            self.adj_value, [self.N, self.N])

            self.attributes1 = tf.Variable(attribute_matrix[:659574], dtype=tf.float32)
            self.attributes2 = tf.Variable(np.zeros((500, 100)), dtype=tf.float32)
            self.attributes = tf.concat([self.attributes1, self.attributes2], axis=0)
            self.attributes_clean = tf.Variable(attribute_matrix, dtype=tf.float32)
            self.attributes_dropout_clean = tf.nn.dropout(self.attributes_clean, rate=dropout)
            self.attributes_dropout = tf.nn.dropout(self.attributes, rate=dropout)

            self.w_init = slim.xavier_initializer
            if gpu_id is None:
                config = tf.ConfigProto(
                    device_count={'GPU': 0}
                )
            else:
                gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)

            session = tf.Session(config=config)
            self.session = session
            self.initializer = tf.local_variables_initializer()

            self.build_v0(drop=dropout)
            self.build_v1(drop=dropout)
            self.build_v2(drop=dropout)
            self.build_v3(drop=dropout)
            # self.build_v4(drop=dropout)
            # self.build_v5(drop=dropout)
            self.attack_build()

    def build_v0(self, with_relu=True, hidden_sizes=[128], drop=0.1):
        with self.graph.as_default():
            self.weights = []
            self.biases = []

            previous_size = self.D
            for ix, layer_size in enumerate(hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}", shape=[previous_size * 2, layer_size], dtype=tf.float32,
                                         initializer=self.w_init())
                bias = tf.get_variable(f"b_{ix + 1}", shape=[layer_size], dtype=tf.float32,
                                       initializer=self.w_init())
                self.weights.append(weight)
                self.biases.append(bias)
                previous_size = layer_size

            weight_final = tf.get_variable(f"W_{len(hidden_sizes) + 1}", shape=[previous_size * 2, self.K],
                                           dtype=tf.float32,
                                           initializer=self.w_init())
            bias_final = tf.get_variable(f"b_{len(hidden_sizes) + 1}", shape=[self.K], dtype=tf.float32,
                                         initializer=self.w_init())
            self.weights.append(weight_final)
            self.biases.append(bias_final)

            hidden = tf.cond(self.training,
                             lambda: self.attributes_dropout_clean,
                             lambda: self.attributes_clean) if drop > 0. else self.attributes_clean
            hidden_holder = tf.cond(self.training,
                             lambda: self.attributes_dropout,
                             lambda: self.attributes) if drop > 0. else self.attributes
            for ix in range(len(hidden_sizes)):
                w = self.weights[ix]
                b = self.biases[ix]

                hidden1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
                hidden = tf.concat([hidden, hidden1], axis=-1)
                hidden = tf.matmul(hidden, w) + b

                hidden3 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder)
                hidden_holder = tf.concat([hidden_holder, hidden3], axis=-1)
                hidden_holder = tf.matmul(hidden_holder, w) + b

                if with_relu:
                    hidden = tf.nn.relu(hidden)
                    hidden_holder = tf.nn.relu(hidden_holder)

                hidden_dropout = tf.nn.dropout(hidden, rate=drop)
                hidden = tf.cond(self.training,
                                 lambda: hidden_dropout,
                                 lambda: hidden) if drop > 0. else hidden
                hidden_holder_dropout = tf.nn.dropout(hidden_holder, rate=drop)
                hidden_holder = tf.cond(self.training,
                                        lambda: hidden_holder_dropout,
                                        lambda: hidden_holder) if drop > 0. else hidden_holder

                if (ix == 0):
                    self.sub_logits = hidden_holder
                    self.sub_logits_clean = hidden
            hidden1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
            hidden = tf.concat([hidden, hidden1], axis=-1)
            self.logits = tf.matmul(hidden, self.weights[-1]) + self.biases[-1]

            hidden2 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder)
            hidden_holder = tf.concat([hidden_holder, hidden2], axis=-1)
            self.logits_holder = tf.matmul(hidden_holder, self.weights[-1]) + self.biases[-1]

            self.logits_gather = tf.gather(self.logits, self.idx[:150000])
            labels_gather = tf.gather(self.true_label_onehot, self.idx[:150000])
            self.logits_gather1 = tf.gather(self.logits, self.val_idx)
            labels_gather1 = tf.gather(self.true_label_onehot, self.val_idx)
            acc_count = tf.equal(tf.argmax(tf.nn.softmax(self.logits_gather1), axis=-1),
                                 tf.argmax(labels_gather1, axis=-1))
            acc_count = tf.cast(acc_count, tf.float32)
            self.acc = tf.reduce_mean(acc_count)

            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather, logits=self.logits_gather)
            self.loss += self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in [self.weights[0], self.biases[0], self.weights[1], self.biases[1]]])

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, var_list=[*self.weights, *self.biases])

    def build_v1(self, with_relu=True, hidden_sizes=[64], drop=0.1):
        with self.graph.as_default():
            w_init = slim.xavier_initializer
            self.weights_v1 = []
            self.biases_v1 = []

            previous_size = self.D
            for ix, layer_size in enumerate(hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}_v1", shape=[previous_size, layer_size], dtype=tf.float32,
                                         initializer=w_init())
                bias = tf.get_variable(f"b_{ix + 1}_v1", shape=[layer_size], dtype=tf.float32,
                                       initializer=w_init())
                self.weights_v1.append(weight)
                self.biases_v1.append(bias)
                previous_size = layer_size

            weight_final = tf.get_variable(f"W_{len(hidden_sizes) + 1}_v1", shape=[previous_size, self.K],
                                           dtype=tf.float32,
                                           initializer=w_init())
            bias_final = tf.get_variable(f"b_{len(hidden_sizes) + 1}_v1", shape=[self.K], dtype=tf.float32,
                                         initializer=w_init())

            self.weights_v1.append(weight_final)
            self.biases_v1.append(bias_final)
            hidden = tf.cond(self.training,
                             lambda: self.attributes_dropout_clean,
                             lambda: self.attributes_clean) if drop > 0. else self.attributes_clean
            hidden_holder = tf.cond(self.training,
                                    lambda: self.attributes_dropout,
                                    lambda: self.attributes) if drop > 0. else self.attributes

            layers = []
            for ix in range(len(hidden_sizes)):
                w = self.weights_v1[ix]
                b = self.biases_v1[ix]

                hidden = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
                hidden = hidden @ w + b
                hidden_holder = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder @ w) + b

                if with_relu:
                    hidden = tf.nn.relu(hidden)
                    hidden_holder = tf.nn.relu(hidden_holder)

                hidden_dropout = tf.nn.dropout(hidden, rate=drop)
                hidden = tf.cond(self.training,
                                 lambda: hidden_dropout,
                                 lambda: hidden) if drop > 0. else hidden
                hidden_dropout_holder = tf.nn.dropout(hidden_holder, rate=drop)
                hidden_holder = tf.cond(self.training,
                                        lambda: hidden_dropout_holder,
                                        lambda: hidden_holder) if drop > 0. else hidden_holder
                layers.append(hidden)
                if (ix == 0):
                    self.sub_logits_v1 = hidden_holder
                    self.sub_logits_v1_clean = hidden

            # hidden = tf.stack(layers, axis=0)
            # hidden = tf.reduce_max(hidden, axis=0)
            hidden = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
            self.logits_v1 = tf.matmul(hidden, self.weights_v1[-1]) + self.biases_v1[-1]

            hidden_holder = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder)
            self.logits_holder_v1 = tf.matmul(hidden_holder, self.weights_v1[-1]) + self.biases_v1[-1]

            # self.logits = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden @ self.weights[-1]) + self.biases[-1]
            self.logits_gather_v1 = tf.gather(self.logits_v1, self.idx[150000:300000])
            labels_gather = tf.gather(self.true_label_onehot, self.idx[150000:300000])
            self.logits_gather1_v1 = tf.gather(self.logits_v1, self.val_idx)
            labels_gather1 = tf.gather(self.true_label_onehot, self.val_idx)
            acc_count = tf.equal(tf.argmax(tf.nn.softmax(self.logits_gather1_v1), axis=-1),
                                 tf.argmax(labels_gather1, axis=-1))
            acc_count = tf.cast(acc_count, tf.float32)
            self.acc_v1 = tf.reduce_mean(acc_count)

            self.loss_v1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather,
                                                                      logits=self.logits_gather_v1)
            self.loss_v1 += self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in
                                                          [self.weights_v1[0], self.biases_v1[0], self.weights_v1[1],
                                                           self.biases_v1[1]]])

            self.optimizer_v1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op_v1 = self.optimizer_v1.minimize(self.loss_v1, var_list=[*self.weights_v1, *self.biases_v1])

    def build_v2(self, with_relu=False, hidden_sizes=[128], drop=0.1):
        with self.graph.as_default():
            w_init = slim.xavier_initializer
            self.weights_v2 = []
            self.biases_v2 = []

            previous_size = self.D
            for ix, layer_size in enumerate(hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}_v2", shape=[previous_size, layer_size], dtype=tf.float32,
                                         initializer=w_init())
                bias = tf.get_variable(f"b_{ix + 1}_v2", shape=[layer_size], dtype=tf.float32,
                                       initializer=w_init())
                self.weights_v2.append(weight)
                self.biases_v2.append(bias)
                previous_size = layer_size

            weight_final = tf.get_variable(f"W_{len(hidden_sizes) + 1}_v2", shape=[previous_size, self.K],
                                           dtype=tf.float32,
                                           initializer=w_init())
            bias_final = tf.get_variable(f"b_{len(hidden_sizes) + 1}_v2", shape=[self.K], dtype=tf.float32,
                                         initializer=w_init())

            self.weights_v2.append(weight_final)
            self.biases_v2.append(bias_final)
            hidden = tf.cond(self.training,
                             lambda: self.attributes_dropout_clean,
                             lambda: self.attributes_clean) if drop > 0. else self.attributes_clean
            hidden_holder = tf.cond(self.training,
                                    lambda: self.attributes_dropout,
                                    lambda: self.attributes) if drop > 0. else self.attributes

            layers = []
            for ix in range(len(hidden_sizes)):
                w = self.weights_v2[ix]
                b = self.biases_v2[ix]

                hidden = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
                # hidden += hidden1
                hidden = hidden @ w + b
                hidden_holder = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder @ w) + b

                if with_relu:
                    hidden = tf.nn.relu(hidden)
                    hidden_holder = tf.nn.relu(hidden_holder)

                hidden_dropout = tf.nn.dropout(hidden, rate=drop)
                hidden = tf.cond(self.training,
                                 lambda: hidden_dropout,
                                 lambda: hidden) if drop > 0. else hidden
                hidden_dropout_holder = tf.nn.dropout(hidden_holder, rate=drop)
                hidden_holder = tf.cond(self.training,
                                        lambda: hidden_dropout_holder,
                                        lambda: hidden_holder) if drop > 0. else hidden_holder
                layers.append(hidden)
                if (ix == 0):
                    self.sub_logits_v2 = hidden_holder
                    self.sub_logits_v2_clean = hidden
                # mean,variance = tf.nn.moments(hidden,axes=1,keep_dims=True)
                # hidden=(hidden-mean)/variance

            # hidden = tf.stack(layers, axis=0)
            # hidden = tf.reduce_max(hidden, axis=0)
            hidden = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
            self.logits_v2 = tf.matmul(hidden, self.weights_v2[-1]) + self.biases_v2[-1]

            hidden_holder = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder)
            self.logits_holder_v2 = tf.matmul(hidden_holder, self.weights_v2[-1]) + self.biases_v2[-1]

            # self.logits = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden @ self.weights[-1]) + self.biases[-1]
            self.logits_gather_v2 = tf.gather(self.logits_v2, self.idx[300000:450000])
            labels_gather = tf.gather(self.true_label_onehot, self.idx[300000:450000])
            self.logits_gather1_v2 = tf.gather(self.logits_v2, self.val_idx)
            labels_gather1 = tf.gather(self.true_label_onehot, self.val_idx)
            acc_count = tf.equal(tf.argmax(tf.nn.softmax(self.logits_gather1_v2), axis=-1),
                                 tf.argmax(labels_gather1, axis=-1))
            acc_count = tf.cast(acc_count, tf.float32)
            self.acc_v2 = tf.reduce_mean(acc_count)

            self.loss_v2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather,
                                                                      logits=self.logits_gather_v2)
            self.loss_v2 += self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in
                                                          [self.weights_v2[0], self.biases_v2[0], self.weights_v2[1],
                                                           self.biases_v2[1]]])

            self.optimizer_v2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op_v2 = self.optimizer_v2.minimize(self.loss_v2, var_list=[*self.weights_v2, *self.biases_v2])

    def build_v3(self, with_relu=False, hidden_sizes=[128], drop=0.1):
        with self.graph.as_default():
            w_init = slim.xavier_initializer
            self.weights_v3 = []
            self.biases_v3 = []

            previous_size = self.D
            for ix, layer_size in enumerate(hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}_v3", shape=[previous_size * 2, layer_size], dtype=tf.float32,
                                         initializer=w_init())
                bias = tf.get_variable(f"b_{ix + 1}_v3", shape=[layer_size], dtype=tf.float32,
                                       initializer=w_init())
                self.weights_v3.append(weight)
                self.biases_v3.append(bias)
                previous_size = layer_size

            weight_final = tf.get_variable(f"W_{len(hidden_sizes) + 1}_v3", shape=[previous_size * 2, self.K],
                                           dtype=tf.float32,
                                           initializer=w_init())
            bias_final = tf.get_variable(f"b_{len(hidden_sizes) + 1}_v3", shape=[self.K], dtype=tf.float32,
                                         initializer=w_init())

            self.weights_v3.append(weight_final)
            self.biases_v3.append(bias_final)
            hidden = tf.cond(self.training,
                             lambda: self.attributes_dropout_clean,
                             lambda: self.attributes_clean) if drop > 0. else self.attributes_clean
            hidden_holder = tf.cond(self.training,
                                    lambda: self.attributes_dropout,
                                    lambda: self.attributes) if drop > 0. else self.attributes

            layers = []
            for ix in range(len(hidden_sizes)):
                w = self.weights_v3[ix]
                b = self.biases_v3[ix]

                hidden1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
                hidden = tf.concat([hidden, hidden1], axis=-1)
                hidden = tf.matmul(hidden, w) + b

                hidden3 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder)
                hidden_holder = tf.concat([hidden_holder, hidden3], axis=-1)
                hidden_holder = tf.matmul(hidden_holder, w) + b

                if with_relu:
                    hidden = tf.nn.relu(hidden)
                    hidden_holder = tf.nn.relu(hidden_holder)

                hidden_dropout = tf.nn.dropout(hidden, rate=drop)
                hidden = tf.cond(self.training,
                                 lambda: hidden_dropout,
                                 lambda: hidden) if drop > 0. else hidden
                hidden_dropout_holder = tf.nn.dropout(hidden_holder, rate=drop)
                hidden_holder = tf.cond(self.training,
                                        lambda: hidden_dropout_holder,
                                        lambda: hidden_holder) if drop > 0. else hidden_holder
                layers.append(hidden)
                if (ix == 0):
                    self.sub_logits_v3_clean = hidden
                    self.sub_logits_v3 = hidden_holder

            # hidden = tf.stack(layers, axis=0)
            # hidden = tf.reduce_max(hidden, axis=0)
            hidden1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
            hidden = tf.concat([hidden, hidden1], axis=-1)
            self.logits_v3 = tf.matmul(hidden, self.weights_v3[-1]) + self.biases_v3[-1]

            hidden_holder1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder)
            hidden_holder = tf.concat([hidden_holder, hidden_holder1], axis=-1)
            self.logits_holder_v3 = tf.matmul(hidden_holder, self.weights_v3[-1]) + self.biases_v3[-1]

            # self.logits = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden @ self.weights[-1]) + self.biases[-1]
            self.logits_gather_v3 = tf.gather(self.logits_v3, self.idx[450000:600000])
            labels_gather = tf.gather(self.true_label_onehot, self.idx[450000:600000])
            self.logits_gather1_v3 = tf.gather(self.logits_v3, self.val_idx)
            labels_gather1 = tf.gather(self.true_label_onehot, self.val_idx)
            acc_count = tf.equal(tf.argmax(tf.nn.softmax(self.logits_gather1_v3), axis=-1),
                                 tf.argmax(labels_gather1, axis=-1))
            acc_count = tf.cast(acc_count, tf.float32)
            self.acc_v3 = tf.reduce_mean(acc_count)

            self.loss_v3 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather,
                                                                      logits=self.logits_gather_v3)
            self.loss_v3 += self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in
                                                          [self.weights_v3[0], self.biases_v3[0], self.weights_v3[1],
                                                           self.biases_v3[1]]])

            self.optimizer_v3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op_v3 = self.optimizer_v3.minimize(self.loss_v3, var_list=[*self.weights_v3, *self.biases_v3])

    def build_v4(self, with_relu=True, hidden_sizes=[64], drop=0.1):
        with self.graph.as_default():
            w_init = slim.xavier_initializer
            self.weights_v4 = []
            self.biases_v4 = []

            previous_size = self.D
            for ix, layer_size in enumerate(hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}_v4", shape=[previous_size * 2, layer_size], dtype=tf.float32,
                                         initializer=w_init())
                bias = tf.get_variable(f"b_{ix + 1}_v4", shape=[layer_size], dtype=tf.float32,
                                       initializer=w_init())
                self.weights_v4.append(weight)
                self.biases_v4.append(bias)
                previous_size = layer_size

            weight_final = tf.get_variable(f"W_{len(hidden_sizes) + 1}_v4", shape=[previous_size * 2, self.K],
                                           dtype=tf.float32,
                                           initializer=w_init())
            bias_final = tf.get_variable(f"b_{len(hidden_sizes) + 1}_v4", shape=[self.K], dtype=tf.float32,
                                         initializer=w_init())

            self.weights_v4.append(weight_final)
            self.biases_v4.append(bias_final)
            hidden = tf.cond(self.training,
                             lambda: self.attributes_dropout_clean,
                             lambda: self.attributes_clean) if drop > 0. else self.attributes_clean
            hidden_holder = tf.cond(self.training,
                                    lambda: self.attributes_dropout,
                                    lambda: self.attributes) if drop > 0. else self.attributes

            layers = []
            for ix in range(len(hidden_sizes)):
                w = self.weights_v4[ix]
                b = self.biases_v4[ix]

                hidden1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
                hidden = tf.concat([hidden, hidden1], axis=-1)
                hidden = tf.matmul(hidden, w) + b

                hidden3 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder)
                hidden_holder = tf.concat([hidden_holder, hidden3], axis=-1)
                hidden_holder = tf.matmul(hidden_holder, w) + b

                if with_relu:
                    hidden = tf.nn.relu(hidden)
                    hidden_holder = tf.nn.relu(hidden_holder)

                hidden_dropout = tf.nn.dropout(hidden, rate=drop)
                hidden = tf.cond(self.training,
                                 lambda: hidden_dropout,
                                 lambda: hidden) if drop > 0. else hidden
                hidden_dropout_holder = tf.nn.dropout(hidden_holder, rate=drop)
                hidden_holder = tf.cond(self.training,
                                        lambda: hidden_dropout_holder,
                                        lambda: hidden_holder) if drop > 0. else hidden_holder
                layers.append(hidden)
                if (ix == 0):
                    self.sub_logits_v4_clean = hidden
                    self.sub_logits_v4 = hidden_holder

            # hidden = tf.stack(layers, axis=0)
            # hidden = tf.reduce_max(hidden, axis=0)
            hidden1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
            hidden = tf.concat([hidden, hidden1], axis=-1)
            self.logits_v4 = tf.matmul(hidden, self.weights_v4[-1]) + self.biases_v4[-1]

            hidden_holder1 = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder)
            hidden_holder = tf.concat([hidden_holder, hidden_holder1], axis=-1)
            self.logits_holder_v4 = tf.matmul(hidden_holder, self.weights_v4[-1]) + self.biases_v4[-1]

            # self.logits = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden @ self.weights[-1]) + self.biases[-1]
            self.logits_gather_v4 = tf.gather(self.logits_v4, self.idx[450000:600000])
            labels_gather = tf.gather(self.true_label_onehot, self.idx[450000:600000])
            self.logits_gather1_v4 = tf.gather(self.logits_v4, self.val_idx)
            labels_gather1 = tf.gather(self.true_label_onehot, self.val_idx)
            acc_count = tf.equal(tf.argmax(tf.nn.softmax(self.logits_gather1_v4), axis=-1),
                                 tf.argmax(labels_gather1, axis=-1))
            acc_count = tf.cast(acc_count, tf.float32)
            self.acc_v4 = tf.reduce_mean(acc_count)

            self.loss_v4 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather,
                                                                      logits=self.logits_gather_v4)
            self.loss_v4 += self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in
                                                          [self.weights_v4[0], self.biases_v4[0], self.weights_v4[1],
                                                           self.biases_v4[1]]])

            self.optimizer_v4 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op_v4 = self.optimizer_v4.minimize(self.loss_v4, var_list=[*self.weights_v4, *self.biases_v4])

    def build_v5(self, with_relu=True, hidden_sizes=[64], drop=0.1):
        with self.graph.as_default():
            w_init = slim.xavier_initializer
            self.weights_v5 = []
            self.biases_v5 = []

            previous_size = self.D
            for ix, layer_size in enumerate(hidden_sizes):
                weight = tf.get_variable(f"W_{ix + 1}_v5", shape=[previous_size, layer_size], dtype=tf.float32,
                                         initializer=w_init())
                bias = tf.get_variable(f"b_{ix + 1}_v5", shape=[layer_size], dtype=tf.float32,
                                       initializer=w_init())
                self.weights_v5.append(weight)
                self.biases_v5.append(bias)
                previous_size = layer_size

            weight_final = tf.get_variable(f"W_{len(hidden_sizes) + 1}_v5", shape=[previous_size, self.K],
                                           dtype=tf.float32,
                                           initializer=w_init())
            bias_final = tf.get_variable(f"b_{len(hidden_sizes) + 1}_v5", shape=[self.K], dtype=tf.float32,
                                         initializer=w_init())

            self.weights_v5.append(weight_final)
            self.biases_v5.append(bias_final)
            hidden = tf.cond(self.training,
                             lambda: self.attributes_dropout_clean,
                             lambda: self.attributes_clean) if drop > 0. else self.attributes_clean
            hidden_holder = tf.cond(self.training,
                                    lambda: self.attributes_dropout,
                                    lambda: self.attributes) if drop > 0. else self.attributes

            layers = []
            for ix in range(len(hidden_sizes)):
                w = self.weights_v5[ix]
                b = self.biases_v5[ix]

                hidden = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
                hidden = hidden @ w + b
                hidden_holder = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder @ w) + b

                if with_relu:
                    hidden = tf.nn.relu(hidden)
                    hidden_holder = tf.nn.relu(hidden_holder)

                hidden_dropout = tf.nn.dropout(hidden, rate=drop)
                hidden = tf.cond(self.training,
                                 lambda: hidden_dropout,
                                 lambda: hidden) if drop > 0. else hidden
                hidden_dropout_holder = tf.nn.dropout(hidden_holder, rate=drop)
                hidden_holder = tf.cond(self.training,
                                 lambda: hidden_dropout_holder,
                                 lambda: hidden_holder) if drop > 0. else hidden_holder
                layers.append(hidden)
                if (ix == 0):
                    self.sub_logits_v5 = hidden_holder
                    self.sub_logits_v5_clean = hidden

            # hidden = tf.stack(layers, axis=0)
            # hidden = tf.reduce_max(hidden, axis=0)
            hidden = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden)
            self.logits_v5 = tf.matmul(hidden, self.weights_v5[-1]) + self.biases_v5[-1]

            hidden_holder = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden_holder)
            self.logits_holder_v5 = tf.matmul(hidden_holder, self.weights_v5[-1]) + self.biases_v5[-1]

            # self.logits = tf.sparse_tensor_dense_matmul(self.adj_norm, hidden @ self.weights[-1]) + self.biases[-1]
            self.logits_gather_v5 = tf.gather(self.logits_v5, self.idx[150000:300000])
            labels_gather = tf.gather(self.true_label_onehot, self.idx[150000:300000])
            self.logits_gather1_v5 = tf.gather(self.logits_v5, self.val_idx)
            labels_gather1 = tf.gather(self.true_label_onehot, self.val_idx)
            acc_count = tf.equal(tf.argmax(tf.nn.softmax(self.logits_gather1_v5), axis=-1),
                                 tf.argmax(labels_gather1, axis=-1))
            acc_count = tf.cast(acc_count, tf.float32)
            self.acc_v5 = tf.reduce_mean(acc_count)

            self.loss_v5 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_gather,
                                                                      logits=self.logits_gather_v5)
            self.loss_v5 += self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in
                                                          [self.weights_v5[0], self.biases_v5[0], self.weights_v5[1],
                                                           self.biases_v5[1]]])

            self.optimizer_v5 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op_v5 = self.optimizer_v5.minimize(self.loss_v5, var_list=[*self.weights_v5, *self.biases_v5])

    def attack_build(self):
        with self.graph.as_default():
            logits_attack = tf.gather(self.logits_holder, self.idx_attack)
            logits_attack_v3 = tf.gather(self.logits_holder_v3, self.idx_attack)
            logits_attack_v1 = tf.gather(self.logits_holder_v1, self.idx_attack)
            logits_attack_v2 = tf.gather(self.logits_holder_v2, self.idx_attack)
            # logits_attack_v4 = tf.gather(self.logits_holder_v4, self.idx_attack)
            # logits_attack_v5 = tf.gather(self.logits_holder_v5, self.idx_attack)
            self.false_label = tf.placeholder(tf.float32)
            self.true_label = tf.placeholder(tf.float32)
            labels_atk2 = tf.gather(self.true_label, self.idx_attack)

            logits_attack_ensemble = (
                                                 logits_attack + logits_attack_v3 + logits_attack_v1 + logits_attack_v2) / 4.
            margin = 0.8
            z = tf.nn.softmax(logits_attack_ensemble, axis=-1)

            attack_loss_per_node = tf.minimum(
                tf.reduce_max(z * tf.cast((1 - labels_atk2), tf.float32), axis=-1) - tf.reduce_max(
                    z * tf.cast(labels_atk2, tf.float32), axis=-1), margin)
            self.attack_loss = tf.reduce_mean(attack_loss_per_node)

            a = 0.9
            b = 0.1
            c = 0.999
            d = 0.001
            self.t = tf.assign(self.t, self.t + 1.)
            attr_gradient = tf.gradients(self.attack_loss, self.attributes2)[0]
            self.attr_gradient = attr_gradient / (tf.norm(attr_gradient, axis=1, keepdims=True) + 1e-8)
            self.attr_m = tf.assign(self.attr_m, a * self.attr_m + b * self.attr_gradient)
            # self.attr_update=tf.assign(self.attributes2,tf.clip_by_value(self.attributes2+0.01*self.attr_m,-1,1))
            attr_m = self.attr_m / (1 - tf.pow(a, self.t))
            self.attr_v = tf.assign(self.attr_v, c * self.attr_v + d * self.attr_gradient * self.attr_gradient)
            attr_v = self.attr_v / (1 - tf.pow(c, self.t))
            self.attr_update = tf.assign(self.attributes2,
                                         tf.clip_by_value(self.attributes2 + 0.05 / (
                                                 tf.sqrt(attr_v) + self.eps) * attr_m, -0.88, 0.88))

    def train(self, idx_train, idx_val, n_iters=200, initialize=True, display=True):
        with self.graph.as_default():
            if initialize:
                self.session.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()

            _iter = range(n_iters)
            if display:
                _iter = tqdm(_iter, desc="Training")

            for _it in _iter:
                tt = self.session.run(
                    [self.train_op, self.train_op_v1, self.train_op_v2, self.train_op_v3, self.acc, self.acc_v3,
                     self.acc_v1, self.acc_v2],
                    feed_dict={self.idx: idx_train, self.training: True,
                               self.adj_value_origin: self.value2,
                               self.adj_value_add: self.value1,
                               self.sparse_slice: self.cood, self.val_idx: idx_val})
                _iter.set_description("acc: %f %f %f %f" % (tt[4], tt[5], tt[6], tt[7]))
            self.saver.save(self.session, "../temp/gcn.ckpt")

    def get_logits(self):
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(self.session, "../temp/gcn.ckpt")
            tt = self.session.run(
                [self.logits, self.logits_v1, self.logits_v2, self.logits_v3],
                feed_dict={self.training: False,
                           self.adj_value_origin: self.value2,
                           self.adj_value_add: self.value1,
                           self.sparse_slice: self.cood})
            return (tt[0] + tt[1] + tt[2] + tt[3]) / 4., np.stack(
                [np.eye(18)[tt[0].argmax(1)], np.eye(18)[tt[1].argmax(1)], np.eye(18)[tt[2].argmax(1)],
                 np.eye(18)[tt[3].argmax(1)]])

    def attack(self, idx_attack, split_train):
        print("idx attack", np.min(idx_attack), np.max(idx_attack))
        attack_iters = 200
        flags = tf.flags
        FLAGS = flags.FLAGS
        filename = '../temp/0719' + FLAGS.gpu
        try:
            os.system("mkdir %s" % filename)
        except:
            print("error")
        with self.graph.as_default():
            saver = tf.train.Saver()
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            saver.restore(self.session, "../temp/gcn.ckpt")
            numline = 25
            for j in range(attack_iters):
                for i in range(0, 500, numline):
                    print("current iteration %d, node %d" % (j, i))
                    cood, value, shape = utils.sparse_to_tuple1(utils.preprocess_graph(self.adjacency_matrix))
                    if (j == 0 and i == 0):
                        train = True
                    else:
                        train = False
                    _loss, _attribute, attr_gradient = self.session.run(
                        [self.attack_loss, self.attr_update, self.attr_gradient],
                        feed_dict={self.idx_attack: idx_attack, self.idx: split_train,
                                   self.true_label: self.true_label_onehot,
                                   self.false_label: self.false_label_onehot,
                                   self.adj_value_origin:
                                       value[2:],
                                   self.adj_value_add: value[:2],
                                   self.sparse_slice: cood,
                                   self.training: train,
                                   })
                    print("loss", _loss)
                    # print(np.max(adj_gradient_norm),np.min(adj_gradient_norm))
                    print(np.max(_attribute), np.min(_attribute), np.mean(np.abs(_attribute)))
                    print(np.min(attr_gradient), np.max(attr_gradient))

                    maxd = self.adjacency_matrix[659574:].getnnz(axis=1).max()
                    print(maxd)
                    assert maxd <= 100
                if ((j % 1 == 0 or j == attack_iters - 1) and j >= 5):
                    if (j == 6):
                        with open("%s/adj_%d.pkl" % (filename, j), "wb+") as f:
                            pickle.dump(self.adjacency_matrix, f)
                    np.save("%s/feature_%d.npy" % (filename, j), _attribute)
        return self.adjacency_matrix, _attribute
