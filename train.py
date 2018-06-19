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

