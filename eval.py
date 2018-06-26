# -*- coding:utf-8 -*-

import tensorflow as tf
import os


from read_data import *
from utils import *
from ops import *
from model import *
from model import BATCH_SIZE


def eval():
    #
    test_dir = './eval/'

    checkpoint_dir = './logs'
    y = tf.placeholder(tf.float32, [BATCH_SIZE,10], name='y')
    z = tf.placeholder(tf.float32, [None, 100], name='z')

    G = generator(z,y)
    data_x, data_y = read_data()
    sample_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
    sample_labels = data_y[120:184]

    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    #saver
    saver = tf.train.Saver(tf.all_variables())

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config = config)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

    test_sess = sess.run(G, feed_dict={z: sample_z, y:sample_labels})

    save_image(test_sess, [8,8], test_dir +'test_%d.png' % 500)

    sess.close()

if __name__=='__main__':
    eval()