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
                mean, variance = tf.identity(batch_mean), tf.identity(batch_variance)  #tf.identity 是一个op操作
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_normalization((value, mean, variance, beta, gamma, 1e-5))

        else:
            mean = bias('moving_mean', [shape], 0.0, False)
            variance = bias('moving_variance',[shape], 1.0, False)

            return tf.nn.batch_normalization(value, mean, variance, beta, gamma, epsilon)


def generator(z, is_train=True, name='generator'):
    with tf.name_scope(name):
        s2, s4, s8, s16 = OUTPUT_SIZE/2, OUTPUT_SIZE/4, OUTPUT_SIZE/8, OUTPUT_SIZE/16
        h1 = tf.reshape(fully_connected(z, GF*8*s16*s16, 'g_fc1'), [-1, s16, s16, GF*8], name='reshap')
        h1 = relu(batch_norm(h1, name='g_bn1', is_train=is_train))
        h2 = deconv2d(h1, [BATCH_SIZE, s8, s4, GF*2], name='g_deconv2d1')
        h2 = relu(batch_norm(h2, name='g_bn2', is_train=is_train))
        h3 = deconv2d(h2, [BATCH_SIZE, s4, s4, GF*2], name='g_deconv2d2')
        h3 = relu(batch_norm(h3, name='g_bn3', is_train=is_train))
        h4 = deconv2d(h3, [BATCH_SIZE, s2, s2, GF*1], name='g_deconv2d3')
        h4 = relu(batch_norm(h4, name='g_bn4', is_train=is_train))
        h5 = deconv2d(h4, [BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 3], name='g_deconv2d4')

        return tf.nn.tanh(h5)


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

def sampler(z, is_train=False, name='sampler'):
    with tf.name_scope(name):
        tf.get_variable_scope().reuse_variables()
        return generator(z, is_train= is_train)


def read_and_decode(filename_queue):
    """
    Reads and decode tfrecords
    :param filename_queue:
    :return:
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,features={'image_raw':tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [OUTPUT_SIZE, OUTPUT_SIZE, 3])
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    return image


def inputs(data_dir, batch_size, name='input'):
    """
    Reads input data num_epochs times
    :param data_dir:
    :param batch_size:
    :param name:
    :return:
    """
    with tf.name_scope(name):
        filenames = [os.path.join(data_dir, 'train%d.tfrecords' %ii) for ii in range(12)]
        filename_queue = tf.train.string_input_producer(filenames)
        image = read_and_decode(filename_queue)
        images = tf.train.shuffle_batch([image], batch_size=batch_size, num_threads=4, capacity=20000 + 3*batch_size,
                                        min_after_dequeue=20000)

        return images


def save_images(images, size, path):
    img = (images+1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h*size[0], w*size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w, :] = image

    return scipy.misc.imsave(path, merge_img)


def train():
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_dir = CURRENT_DIR + '/logs_without_condition/'
    data_dir = CURRENT_DIR + '/data/img_align_celeba_tfrecords/'

    images = inputs(data_dir, BATCH_SIZE)

    z = tf.placeholder(tf.float32, [None, Z_DIM], name='z')

    G = generator(z)
    D, D_logits = discriminator(images)
    samples = sampler(z)
    D_, D_logits_ = discriminator(G, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_)))

    z_sum = tf.summary.histogram('z', z)
    d_sum = tf.summary.histogram('d', D)
    d__sum = tf.summary.histogram('d_', D_)
    G_sum = tf.summary.histogram('G', G)

    d_loss_real_sum = tf.summary.scalar('d_loss_real', d_loss_real)
    d_loss_fake_sum = tf.summary.scalar('d_loss_fake', d_loss_fake)
    d_loss_sum = tf.summary.scalar('d_loss', d_loss)
    g_loss_sum = tf.summary.scalar('g_loss', g_loss)

    g_sum = tf.summary.merge([z_sum, d__sum, G_sum, d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    saver = tf.train.Saver()

    d_optim = tf.train.AdamOptimizer(LR, beta1=0.5) \
        .minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optim = tf.train.AdamOptimizer(LR, beta1=0.5) \
        .minimize(g_loss, var_list=g_vars, global_step=global_step)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)

    writer = tf.summary.FileWriter(train_dir, sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init = tf.initialize_all_variables()
    sess.run(init)

    start = 0
    if LOAD_MODEL:
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(train_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(train_dir, ckpt_name))
            global_step = ckpt.model_checkpoint_path.split('/')[-1] \
                .split('-')[-1]
            print('Loading success, global_step is %s' % global_step)

        start = int(global_step)

    for epoch in range(EPOCH):

        batch_idxs = 3072

        if epoch:
            start = 0

        for idx in range(start, batch_idxs):

            batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM))

            _, summary_str = sess.run([d_optim, d_sum], feed_dict={z: batch_z})
            writer.add_summary(summary_str, idx + 1)

            # Update G network
            _, summary_str = sess.run([g_optim, g_sum], feed_dict={z: batch_z})
            writer.add_summary(summary_str, idx + 1)

            # Run g_optim twice to make sure that d_loss does not go to zero
            _, summary_str = sess.run([g_optim, g_sum], feed_dict={z: batch_z})
            writer.add_summary(summary_str, idx + 1)

            errD_fake = d_loss_fake.eval({z: batch_z})
            errD_real = d_loss_real.eval()
            errG = g_loss.eval({z: batch_z})
            if idx % 20 == 0:
                print("[%4d/%4d] d_loss: %.8f, g_loss: %.8f" \
                      % (idx, batch_idxs, errD_fake + errD_real, errG))

            if idx % 100 == 0:
                sample = sess.run(samples, feed_dict={z: sample_z})
                samples_path = CURRENT_DIR + '/samples_without_condition/'
                save_images(sample, [8, 8],
                            samples_path + \
                            'sample_%d_epoch_%d.png' % (epoch, idx))

                print('\n' * 2)
                print('===========    %d_epoch_%d.png save down    ==========='
                      % (epoch, idx))
                print('\n' * 2)

            if (idx % 512 == 0) or (idx + 1 == batch_idxs):
                checkpoint_path = os.path.join(train_dir,
                                               'my_dcgan_tfrecords.ckpt')
                saver.save(sess, checkpoint_path, global_step=idx + 1)
                print('*********    model saved    *********')

        print('******* start with %d *******' % start)

    coord.request_stop()
    coord.join(threads)
    sess.close()


def evaluate():
    eval_dir = CURRENT_DIR + '/eval/'

    checkpoint_dir = CURRENT_DIR + '/logs_without_condition/'

    z = tf.placeholder(tf.float32, [None, Z_DIM], name='z')

    G = generator(z, is_train=False)

    sample_z1 = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM))
    sample_z2 = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM))
    sample_z3 = (sample_z1 + sample_z2) / 2
    sample_z4 = (sample_z1 + sample_z3) / 2
    sample_z5 = (sample_z2 + sample_z3) / 2

    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    saver = tf.train.Saver(tf.all_variables())

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print('Loading success, global_step is %s' % global_step)

    eval_sess1 = sess.run(G, feed_dict={z: sample_z1})
    eval_sess2 = sess.run(G, feed_dict={z: sample_z4})
    eval_sess3 = sess.run(G, feed_dict={z: sample_z3})
    eval_sess4 = sess.run(G, feed_dict={z: sample_z5})
    eval_sess5 = sess.run(G, feed_dict={z: sample_z2})

    print(eval_sess3.shape)

    save_images(eval_sess1, [8, 8], eval_dir + 'eval_%d.png' % 1)
    save_images(eval_sess2, [8, 8], eval_dir + 'eval_%d.png' % 2)
    save_images(eval_sess3, [8, 8], eval_dir + 'eval_%d.png' % 3)
    save_images(eval_sess4, [8, 8], eval_dir + 'eval_%d.png' % 4)
    save_images(eval_sess5, [8, 8], eval_dir + 'eval_%d.png' % 5)

    sess.close()


if __name__ == '__main__':

    if TRAIN:
        train()
    else:
        evaluate()
