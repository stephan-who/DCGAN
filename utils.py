import scipy.misc
import numpy as np

#保存图片
def save_image(images, size, path):
    """

    :param images:
    :param size:
    :param path:
    :return:
    """

    img = (images + 1.0 ) /2.0
    h, w = img.shape[1], img.shape[2]

    #产生一个画布，用来保存生成的batch_size个图像
    merge_img = np.zeros((h*size[0], w*size[1],3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx //size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w, :] = image

    return scipy.misc.imsave(path, merge_img)


