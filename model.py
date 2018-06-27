import tensorflow as tf
from ops import *

BATCH_SIZE = 64

#定义生成器
def generator(z, y, train = True):
    #y 是一个【BATCH_SIZE, 10]维的向量， 把y转成四维张量
    yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 10], name='yb')
    #
    z = tf.concat([z, y],1, name = 'z_concat_y')
    #
    h1 = tf.nn.relu(batch_norm_layer(fully_connected(z, 1024, 'g_fully_connected1'), is_train=train, name='g_bn1'))

    h1 = tf.concat( [h1, y],1, name='active1_concat_y')
    h2 = tf.nn.relu(batch_norm_layer(fully_connected(h1, 128 * 49), 'g_fully_connected2'), name='g_bn2')
    h2 = tf.reshape(h2, [64, 7,7,128], name='h2_reshape')
    h2 = conv_cond_concat(h2, yb, name='activate2_concat_y')
    h3 = tf.nn.relu(batch_norm_layer(deconv2d(h2, [64,14,14,128], name='g_deconv2d3'), is_train=train, name='g_bn3'))
    h3 = conv_cond_concat(h3, yb, name='active3_concat_y')

    #sigmoid,把值归一化维0-1之间
    h4 = tf.nn.sigmoid(deconv2d(h3, [64,28,28,1], name='g_deconv2d4'), name='generate_image')

    return h4


#define discriminator
def discriminator(image, y, reuse=False):
    #因为真实数据和生成数据都要经过判别器， 所以需要制定reuse是否可用
    if reuse:
        tf.get_variable_scope().reuse_variables()

    yb = tf.reshape(y, [BATCH_SIZE, 1,1,10],name='yb')
    x = conv_cond_concat(image, yb, name='image_concat_y')
    #卷积，激活，串联条件
    h1 = lrelu(conv2d(x,11, name='d_conv2d1'), name='lrelu1')
    h1 = conv_cond_concat(h1, yb, name='h1_concat_yb')
    h2 = lrelu(batch_norm_layer(conv2d(h1, 74, name='d_conv2d2'),name = 'd_bn2'),name='lrelu2')
    h2 = tf.reshape(h2, [BATCH_SIZE, -1],name='reshap_lrelu2_to_2d')
    h2 = tf.concat([h2,y],1, name='lrelu_concat_y')
    h3 = lrelu(batch_norm_layer(fully_connected(h2, 1024, name='d_fully_connected3'), name='d_bn3'),name='lrelu3')
    h3 = tf.concat([h3,y],1,name='lrelu3_concat_y')
    #全连接层，输出以为loss值
    h4 = fully_connected(h3, 1, name='d_result_withouts_sigmoid')

    return tf.nn.sigmoid(h4, name='discriminator_result_with_sigmoid'), h4

#定义训练过程中的采样函数
def sampler(z, y, train=True):
    tf.get_variable_scope().reuse_variables()
    return generator(z, y, train=train)

