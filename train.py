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

    z_sum = tf.summary.histogram("z",z)
    d_sum = tf.summary.histogram("d",D)
    d__sum = tf.summary.histogram("d_", D_)
    G_sum = tf.summary.image("G",G)

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)

    #合并各自的总结
    g_sum = tf.summary.merge([z_sum, d__sum, G_sum, d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])

    #生成器和判别器要更新的变量，由于tf.train.Optimizer的var_list
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    saver = tf.train.Saver()

    #优化算法采用Adam
    d_optim = tf.train.AdamOptimizer(0.0002, beta1= 0.5).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optim = tf.train.AdamOptimizer(0.0002, beta1= 0.5).minimize(g_loss, var_list=g_vars, global_step=global_step)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.2