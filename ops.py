import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm




def bias(name, shape, bias_start=0.0, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable, initializer=tf.constant_initializer(bias_start,
                                                                                                            dtype=dtype))
    return var
    # tf.constant_initializer == tf.Constant()
    # tf.zeros_initializer() == tf.Zeros()
    # tf.ones_initializer() == tf.Ones()


def weight(name, shape, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable, initializer=tf.random_normal_initializer(
        stddev=stddev, dtype=dtype))
    return var


# 全连接层
def fully_connected(value, output_shape, name='fully_connected', with_w=False):
    shape = value.get_shape().as_list()

    with tf.variable_scope(name):
        weights = weight('weights', [shape[1], output_shape], 0.02)
        biases = bias('biases', [output_shape], 0.0)

    if with_w:
        return tf.matmul(value, weights) + biases, weights, biases
    else:
        return tf.matmul(value, weights) + biases


# leaky-relu
def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x, name=name)


def relu(value, name='relu'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)


def deconv2d(value, output_shape, k_h=5, k_w=5, strides=[1, 2, 2, 1], name='deconv2d', with_w=False):
    with tf.variable_scope(name):
        weights = weight('weights', [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
        # 卷积核的高，卷积核的宽，卷积核个数，图像通道数
        #filter_height, filter_width, out_channels, in_channels
        deconv = tf.nn.conv2d_transpose(value, weights, output_shape, strides=strides)
        biases = bias('biases', [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv


# 卷积层
def conv2d(value, output_dim, k_h=5, k_w=5, strides=[1, 2, 2, 1], name='conv2d'):
    with tf.variable_scope(name):
        weights = weight('weights', [k_h, k_w, value.get_shape()[-1], output_dim])
        #filter_height, filter_width, in_channels, out_channels
        conv = tf.nn.conv2d(value, weights, strides=strides, padding='SAME')
        biases = bias('biases', [output_dim])
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


# 把约束条件串联到featur map
def conv_cond_concat(value, cond, name='concat'):
    # 把张量的纬度形状转化为Python的list
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()

    with tf.variable_scope(name):
        return tf.concat(3, [value, cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])])


# Batch Normalization 层
def batch_norm_layer(value, is_train=True, name='batch_norm'):
    with tf.variable_scope(name) as scope:
        if is_train:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True, is_training=is_train,
                              updates_collections=None, scope=scope)
        else:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True, is_training=is_train, reuse=True,
                              updates_collections=None, scope=scope)
