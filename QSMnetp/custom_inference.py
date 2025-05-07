import tensorflow as tf
import scipy.io
import time
import os
import numpy as np
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

def zero_padding(array, factor=8):
    original_shape = np.array(array.shape)

    new_shape = np.ceil(original_shape / factor) * factor
    new_shape = new_shape.astype(int)

    padding_width = new_shape - original_shape
    pad_before = (padding_width // 2).astype(int)
    pad_after = padding_width - pad_before

    pad_config = [(pad_before[i], pad_after[i]) for i in range(len(original_shape))]

    padded_array = np.pad(array, pad_config, mode='constant', constant_values=0)

    padding_info = {
        'original_shape': original_shape,
        'pad_width': pad_config
    }

    return padded_array, padding_info


def zero_removing(padded_array, padding_info):
    original_shape = padding_info['original_shape']
    pad_width = padding_info['pad_width']

    slices = tuple(slice(pad_width[i][0], pad_width[i][0] + original_shape[i])
                   for i in range(len(original_shape)))

    unpadded_array = padded_array[slices]

    return unpadded_array


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
    pfield, padding_info = zero_padding(field, factor=16)
    N = pfield.shape
    pfield = np.expand_dims(pfield, axis=0)
    pfield = np.expand_dims(pfield, axis=4)

    Z = tf.compat.v1.placeholder("float", [None, N[0], N[1], N[2], 1])
    keep_prob = tf.compat.v1.placeholder("float")

    net_func = getattr(network_model, net_model)
    feed_result = net_func(Z, act_func, False, False)

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        print('##########Restore Network##########')
        saver.restore(sess, dir_net + network_name + '/' + network_name + '-' + str(epoch))
        print('Done!')
        print('##########Inference...##########')
        result_im = y_std * sess.run(feed_result, feed_dict={Z: pfield, keep_prob: 1.0}) + y_mean
        result_im = zero_removing(result_im.squeeze(), padding_info)

    print('All done!')
    return result_im


if __name__ == '__main__':
    start_time = time.time()
    inf()
    print("Total inference time : {} sec".format(time.time() - start_time))





