#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from config import config
from dataset import Dataset
from IPython import embed
import argparse



class Model():
    def __init__(self, loss='softmax'):
        # set the initializer of conv_weight and conv_bias
        self.weight_init = tf_contrib.layers.variance_scaling_initializer(factor=1.0,
                                mode='FAN_IN', uniform=False)
        self.bias_init = tf.zeros_initializer()
        self.reg = tf_contrib.layers.l2_regularizer(config.weight_decay)
        self.placeholder, self.label_onehot, self.logits = self.build()

        if loss == 'softmax':
            self.predict = tf.nn.softmax(self.logits)
        elif loss == 'abs-max':
            abs_logits = tf.abs(self.logits)
            self.predict = abs_logits / tf.reduce_sum(abs_logits, axis=1, keepdims=True)
        elif loss == 'square-max':
            square_logits = self.logits * self.logits
            self.predict = square_logits / tf.reduce_sum(square_logits, axis=1, keepdims=True)
        elif loss == 'plus-one-abs-max':
            plus_one_abs_logits = tf.abs(self.logits) + 1.0
            self.predict = plus_one_abs_logits / tf.reduce_sum(plus_one_abs_logits, axis=1, keepdims=True)
        elif loss == 'non-negative-max':
            relu_logits = tf.nn.relu(self.logits)
            self.predict = relu_logits / tf.reduce_sum(relu_logits, axis=1, keepdims=True)
        else:
            raise("Invalid loss type")

        self.saver = tf.train.Saver()
        self.path = "train_log/models/epoch-52-10140"

    def _conv_layer(self, name, inp, kernel_shape, stride, padding='SAME',is_training=False):
        with tf.variable_scope(name) as scope:
            conv_filter = tf.get_variable(name='filter', shape=kernel_shape,
                                          initializer=self.weight_init, regularizer=self.reg)
            conv_bias = tf.get_variable(name='bias', shape=kernel_shape[-1],
                                        initializer=self.bias_init)
            x = tf.nn.conv2d(inp, conv_filter, strides=[1, stride, stride, 1],
                             padding=padding, data_format='NHWC')
            x = tf.nn.bias_add(x, conv_bias, data_format='NHWC')
            x = tf.layers.batch_normalization(x, axis=3, training=is_training)
            x = tf.nn.relu(x)
        return x

    def _pool_layer(self, name, inp, ksize, stride, padding='SAME', mode='MAX'):
        assert mode in ['MAX', 'AVG'], 'the mode of pool must be MAX or AVG'
        if mode == 'MAX':
            x = tf.nn.max_pool(inp, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                               padding=padding, name=name, data_format='NHWC')
        elif mode == 'AVG':
            x = tf.nn.avg_pool(inp, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                               padding=padding, name=name, data_format='NHWC')
        return x

    def _fc_layer(self, name, inp, units, dropout=0.5):
        with tf.variable_scope(name) as scope:
            shape = inp.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(inp, [-1, dim]) # flatten
            if dropout > 0:
                x = tf.nn.dropout(x, keep_prob=dropout, name='dropout')
            x = tf.layers.dense(x, units, kernel_initializer=self.weight_init,
                                bias_initializer=self.bias_init, kernel_regularizer=self.reg)
        return x

    #def _softmax_layer(self, name, inp):
    #    x = tf.nn.softmax(inp, name=name)
    #    return x

    def build(self):
        data = tf.placeholder(tf.float32, shape=(None,)+config.image_shape+(config.nr_channel,),
                              name='data')
        label = tf.placeholder(tf.int32, shape=(None,), name='label')
        # convert the format of label to one-hot
        label_onehot = tf.one_hot(label, config.nr_class, dtype=tf.int32)
        # a setting for bn
        is_training = tf.placeholder(tf.bool, name='is_training')

        # conv1
        x = self._conv_layer(name='conv1', inp=data,
                             kernel_shape=[3, 3, config.nr_channel, 16], stride=1,
                             is_training=is_training) # Nx32x32x32
        x = self._pool_layer(name='pool1', inp=x, ksize=2, stride=2, mode='MAX') # Nx16x16x16

        # conv2
        x = self._conv_layer(name='conv21', inp=x, kernel_shape=[3, 3, 16, 32],
                             stride=1, is_training=is_training)
        x = self._conv_layer(name='conv22', inp=x, kernel_shape=[3, 3, 32, 32],
                             stride=1, is_training=is_training)
        x = self._pool_layer(name='pool2', inp=x, ksize=2, stride=2, mode='MAX') # Nx8x8x32

        # conv3
        x = self._conv_layer(name='conv31', inp=x, kernel_shape=[3, 3, 32, 64],
                             stride=1, is_training=is_training)
        x = self._conv_layer(name='conv32', inp=x, kernel_shape=[3, 3, 64, 64],
                             stride=1, is_training=is_training)
        x = self._pool_layer(name='pool3', inp=x, ksize=2, stride=2, mode='MAX') # Nx4x4x64

        # fc1
        logits = self._fc_layer(name='fc1', inp=x, units=config.nr_class, dropout=0)

        placeholders = {
            'data': data,
            'label': label,
            'is_training': is_training,
        }
        return placeholders, label_onehot, logits

    def start_session(self):
        self.sess = tf.Session()

    def load_from_checkpoint(self):
        self.saver.restore(self.sess, self.path)

    def get_prediction(self, data):
        return self.sess.run(self.predict, feed_dict={self.placeholder['data']:data, self.placeholder['is_training']: False})

    def close_session(self):
        self.sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loss', default='softmax')
    args = parser.parse_args()
    assert args.loss in ['softmax', 'abs-max', 'square-max', 'plus-one-abs-max', 'non-negative-max']

    teacher_network = Model(args.loss)
    teacher_network.start_session()
    teacher_network.load_from_checkpoint()

    ds = Dataset('train')
    ds = ds.load()
    gen = ds.instance_generator()

    images = []
    t = 0
    for i in range(100):
        img, label = next(gen)
        start = time.time()
        target_labels = teacher_network.get_prediction([img])
        dur = (time.time() - start)
        t += dur
    print('| light nn |', args.loss, '|', round(dur / 100 * 1000, 5), '| ms')
