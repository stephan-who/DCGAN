import tensorflow as tf
import os

from read_data import *
from utils import *
from ops import *
from model import *
from model import BATCH_SIZE


def train():

    global_step = tf.Variable(0, name='global_step', trainable= False)
    train_dir = './logs'

    #z表示随机噪声,y表示约束条件
    y = tf.placeholder(tf.float32, [BATCH_SIZE, 10], name='y')
    images = tf.placeholder(tf.float32, [64,28,28,1], name='real_images')
    z = tf.placeholder(tf.float32, [None, 100], name='z')

    #由生成器生成图像G

    G = generator(z, y)
    #真实图像送入判别器
    D, D_logits = discriminator(images, y)

    samples = sampler(z,y)
    D_, D_logits_ = discriminator(G, y, reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_)))

    z_sum = tf.