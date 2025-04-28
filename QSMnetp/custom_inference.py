import tensorflow as tf
import scipy.io
import time
import os
import numpy as np
# import sys

# Add the current directory and external_code paths to system path
# sys.path.append('.')
# sys.path.append('./external_code/Code')

from utils import padding_data, crop_data
import network_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

'''
Network model
'''
network_name = 'QSMnet+_64'
epoch = 25
sub_num = 1  # number of subjects in testset
voxel_size = [1, 1, 1]  # resolution of subject
dir_net = '../Checkpoints/'


def inf(field):
    f = scipy.io.loadmat(dir_net + network_name + '/' + 'norm_factor_' + network_name + '.mat')
    net_info = np.load(dir_net + network_name + '/network_info_' + network_name + '.npy')
    act_func = net_info[0]
    net_model = net_info[1]

    b_mean = f['input_mean']
    b_std = f['input_std']
    y_mean = f['label_mean']
    y_std = f['label_std']

    tf.compat.v1.reset_default_graph()

    field = (field - b_mean) / b_std
    [pfield, N_difference, N] = padding_data(field)

    Z = tf.compat.v1.placeholder("float", [None, N[0], N[1], N[2], 1])
    keep_prob = tf.compat.v1.placeholder("float")

    print('pirinpimtin', network_model, net_model)
    net_func = getattr(network_model, net_model)
    print('pipisito', Z, act_func)
    feed_result = net_func(Z, act_func, False, False)
    print('poucha lioqui')

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        print('##########Restore Network##########')
        saver.restore(sess, dir_net + network_name + '/' + network_name + '-' + str(epoch))
        print('Done!')
        print('##########Inference...##########')
        result_im = y_std * sess.run(feed_result, feed_dict={Z: pfield, keep_prob: 1.0}) + y_mean
        result_im = crop_data(result_im.squeeze(), N_difference)

    print('All done!')
    return result_im


if __name__ == '__main__':
    start_time = time.time()
    inf()
    print("Total inference time : {} sec".format(time.time() - start_time))





