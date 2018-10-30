# -*-coding:utf-8-*-
# https://github.com/czzyyy/CapsNet-Tensorflow
import tensorflow as tf
import numpy as np
import Capsule.ops as ops
import Capsule.utils as utl


def conv_capsule(input_x, num_outputs=32, vec_len=8, filter_size=9, stride=2, routing=False, name='conv_capsule'):
    b, h, w, v, c = input_x.shape.as_list()
    input_x = tf.reshape(input_x, shape=(b, h, w, v * c))
    with tf.variable_scope(name):
        input_conv = ops.conv2d(input_x, output_num=num_outputs * vec_len, sn=False, stride=stride,
                                filter_size=filter_size, padding='VALID', name='capsule_1')

        # default reshape [batch_size, 1152, 8, 1]
        input_conv = tf.reshape(input_conv, (b, -1, vec_len, 1))
        assert input_conv.shape == [b, 1152, 8, 1]
        if routing:
            # some try
            # [batch_size, h *w * c,1152, 8, 1]
            input_conv_tile = tf.tile(input_conv, [1, h * w * c, 1, 1, 1])
            assert input_conv_tile.shape == [b, h * w * c, 1152, 8, 1]
            # [batch_size, h *w * c, 1152, 8, 1]
            w = tf.get_variable('w', input_conv_tile.shape, tf.float32,
                                tf.truncated_normal_initializer(stddev=0.02))
            # [batch_size, h *w * c, 1152, 8, 1]
            input_conv_tile = tf.multiply(input_conv_tile, w)
            assert input_conv_tile.shape == [b, h * w * c, 1152, 8, 1]
            # [batch_size, h * w * c, 1152, 1, 1]
            routing_b = tf.constant(np.zeros([b, h * w * c, num_outputs, 1, 1], dtype=np.float32))
            assert routing_b.shape == [b, h * w * c, 1152, 1, 1]
            capsules = ops.routing(input_conv_tile, routing_b)
            assert routing_b.shape == [b, 1, 1152, 8, 1]
            capsules = tf.squeeze(capsules, axis=1)
            assert routing_b.shape == [b, 1152, 8, 1]
        else:
            capsules = ops.squash(input_conv)
            assert capsules.shape == [b, 1152, 8, 1]
        return capsules


def fc_capsule(input_x, num_outputs=10, vec_len=16, routing=True, name='fc_capsule'):
    # [batch_size, 1152, 8, 1]
    b, n, v, _ = input_x.shape.as_list()
    assert input_x.shape == [b, 1152, 8, 1]
    with tf.variable_scope(name):
        # [batch_size, 1152, 10, 8, 16]
        w = tf.get_variable('w', [b, n, num_outputs, v, vec_len], tf.float32,
                            tf.truncated_normal_initializer(stddev=0.02))
        assert w.shape == [b, 1152, 10, 8, 16]
        # [batch_size, 1152, 1, 8, 1]
        input_x = tf.reshape(input_x, shape=[b, n, 1, v, 1])
        assert input_x.shape == [b, 1152, 1, 8, 1]
        # [batch_size, 1152, 10, 8, 1]
        input_x_tile = tf.tile(input_x, [1, 1, num_outputs, 1, 1])
        assert input_x_tile.shape == [b, 1152, 10, 8, 1]
        # [batch_size, 1152, 10, 16, 1]
        input_x_u = tf.matmul(tf.transpose(w, perm=[0, 1, 2, 4, 3]), input_x_tile)
        assert input_x_u.shape == [b, 1152, 10, 16, 1]
        if routing:
            # [batch_size, 1152, 10, 1, 1]
            routing_b = tf.constant(np.zeros([b, n, num_outputs, 1, 1], dtype=np.float32))
            assert routing_b.shape == [b, 1152, 10, 1, 1]
            # [batch_size, 1, 10, 16, 1]
            capsules = ops.routing(input_x_u, routing_b)
            assert capsules.shape == [b, 1, 10, 16, 1]
            # [batch_size, 10, 16, 1]
            capsules = tf.squeeze(capsules, axis=1)
            assert capsules.shape == [b, 10, 16, 1]
        else:
            # some try
            # [batch_size, 1, 10, 16, 1]
            capsules = tf.reduce_mean(input_x_u, axis=1)
            assert capsules.shape == [b, 1, 10, 16, 1]
            capsules = ops.squash(capsules)
            assert capsules.shape == [b, 1, 10, 16, 1]
            # [batch_size, 10, 16, 1]
            capsules = tf.squeeze(capsules, axis=1)
            assert capsules.shape == [b, 10, 16, 1]
        return capsules
