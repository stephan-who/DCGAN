import os
import numpy as np
import scipy.misc
import tensorflow as tf

BATCH_SIZE = 64
OUTPUT_SIZE = 64

GF = 64
DF = 64   #dimension of d filters in first conv layer.default [64]

Z_DIM = 100
IMAGE_CHANNEL = 3
LR = 0.0002
EPOCH = 5
LOAD_MODEL = False
TRAIN = True
CURRENT_DIR = os.getcwd()


def bias(name, shape, bias_start=0.0, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name ,shape, tf.float32, trainable=trainable,
                          initializer=tf.constant_initializer(bias_start, dtype=dtype))

    return var

def weight(name, shape, stddev = 0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name ,shape, tf.float32, trainable=trainable,
                          initializer=tf.random_normal_initializer(stddev=stddev, dtype=dtype))
    return var

def fully_connected(value, output_shape, name='fully_connected',with_w = False):
    shape = value.get_shape().as_list()
    with tf.variable_scope(name):
        weights = weight('weights',[shape[1], output_shape],0.02)
        biases = bias('biases', [output_shape],0.0)

    if with_w:
        return tf.matmul(value, weights) + biases, weights, biases
    else:
        return tf.matmul(value, weights) + biases

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x, leak*x, name=name)

def relu(value, name='relue'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)

def deconv2d(value, output_shape, k_h = 5, k_w=5, strides=[1,2,2,1],name='deconv2d', with_w = False):
    with tf.variable_scope(name):
        weights = weight('weights', [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(value, weights, output_shape,strides=strides)
        biases = bias('biases',[output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv

def conv2d(value, output_dim, k_h=5, k_w=5, strides=[1,2,2,1],name='conv2d'):
    with tf.variable_scope(name):
        weights = weight('weights', [k_h,k_w,value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides=strides, padding='SAME')
        biases = bias('biases',[output_dim])
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def conv_cond_concat(value, cond, name='concat'):
    """
    Concatenate conditioning vector on feature map axis.
    :param value:
    :param cond:
    :param name:
    :return:
    """
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()

    with tf.variable_scope(name):
        return tf.concat(3, [value, cond* tf.ones(value_shapes[0:3] + cond_shapes[3:])])

def batch_norm(value, is_train=True, name='batch_norm', epsilon=1e-5, momentum = 0.9):
    with tf.variable_scope(name):
        ema = tf.train.ExponentialMovingAverage(decay =momentum)#滑动平均法
        # 移动平均法是用一组最近的实际数据值来预测未来一期或几期内公司产品的需求量、公司产能等的一种常用方法。移动平均法适用于即期预测。当产品需求既不快速增长也不快速下降，且不存在季节性因素时，移动平均法能有效地消除预测中的随机波动，是非常有用的。移动平均法根据预测时使用的各元素的权重不同
        # 移动平均法是一种简单平滑预测技术，它的基本思想是：根据时间序列资料、逐项推移，依次计算包含一定项数的序时平均值，以反映长期趋势的方法。因此，当时间序列的数值由于受周期变动和随机波动的影响，起伏较大，不易显示出事件的发展趋势时，使用移动平均法可以消除这些因素的影响，显示出事件的发展方向与趋势（即趋势线），然后依趋势线分析预测序列的长期趋势。
        shape = value.get_shape().as_list()[-1]
        beta = bias('beta', [shape], bias_start=0.0)
        gamma = bias('gamma', [shape], bias_start=1.0)

        if is_train:
            batch_mean, batch_variance = tf.nn.moments(value, [0,1,2],name='moments') #计算统计矩，mean 是一阶矩即均值，variance 则是二阶中心矩即方差，axes=[0]表示按列计算；
            moving_mean = bias('moving_mean', [shape], 0.0,False)
            moving_variance = bias('moving_variance', [shape], 1.0, False)

            ema_apply_op = ema.apply([batch_mean, batch_variance])#按顺序传递参数

            assign_mean = moving_mean.assign(ema.average(batch_mean))
            assign_variance = moving_variance.assign(ema.average(batch_variance))
            with tf.control_dependencies([ema_apply_op]):
                mean, variance = tf.identity(batch_mean), tf.identity(batch_variance)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_normalization((value, mean, variance, beta, gamma, 1e-5))

        else:
            mean = bias('moving_mean', [shape], 0.0, False)
            variance = bias('moving_variance',[shape], 1.0, False)

            return tf.nn.batch_normalization(value, mean, variance, beta, gamma, epsilon)


def generator(z, is_train=True, name='generator'):
    with tf.name_scope(name):



def discriminator(image, reuse=False, name='discriminator'):
    with tf.name_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, DF, name='d_h0_conv'),name='d_h0_lrelu')
        h1 = lrelu(batch_norm(conv2d(h0, DF*2, name='d_h1_conv'), name='d_h1_bn'), name='d_h1_lrelu')
        h2 = lrelu(batch_norm(conv2d(h1, DF*4, name='d_h2_conv')))
        h3 = lrelu(conv2d(h2, DF*8, name='d_h3_conv'))
        h4= fully_connected(tf.reshape(h3, [BATCH_SIZE, -1]), 1, 'd_h4_fc')

        return tf.nn.sigmoid(h4), h4