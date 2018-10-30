# -*- coding:utf-8-*-
import tensorflow as tf


def batch_normalizer(x, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm', reuse=False):
    """
    :param x: input feature map
    :param epsilon:
    :param momentum:
    :param train: train or not?
    :param name:
    :param reuse: reuse or not?
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                            scale=True, is_training=train)


def instance_normalizer(x, name='instance_norm', reuse=False):
    """
    :param x:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        batch, height, width, channel = [i for i in x.shape]
        var_shape = [channel]
        # return axes 's mean and variance
        mu, sigma_sq = tf.nn.moments(x, [1, 2], keep_dims=True)
        # shift is beta, scale is alpha in in_norm form
        shift = tf.get_variable('shift', shape=var_shape, initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', shape=var_shape, initializer=tf.ones_initializer())
        epsilon = 1e-3
        normalized = (x-mu)/(sigma_sq + epsilon)**0.5
        return scale * normalized + shift


def full_connect(x, output_num, sn=True, stddev=0.02, bias=0.0, name='full_connect', reuse=False):
    """
    :param x: the input feature map
    :param output_num: the output feature map size
    :param sn: use spectral_norm or not
    :param stddev:
    :param bias:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        shape = x.shape.as_list()
        w = tf.get_variable('w', [shape[1], output_num], tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        if sn:
            w = spectral_norm(w)
        b = tf.get_variable('b', [output_num], tf.float32, tf.constant_initializer(bias))
        return tf.matmul(x, w) + b


def conv2d(x, output_num, sn=True, stride=2, filter_size=5, stddev=0.02, padding='SAME', name='conv2d', reuse=False):
    """
    :param x:
    :param output_num:
    :param sn: use spectral_norm or not
    :param stride:
    :param filter_size:
    :param stddev:
    :param padding:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, in_channels, output_channels]
        shape = x.shape.as_list()
        filter_shape = [filter_size, filter_size, shape[-1], output_num]
        strides_shape = [1, stride, stride, 1]
        w = tf.get_variable('w', filter_shape, tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        if sn:
            w = spectral_norm(w)
        b = tf.get_variable('b', [output_num], tf.float32, tf.constant_initializer(0.0))
        return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=strides_shape, padding=padding), b)


def deconv2d(x, output_size, sn=True, stride=2, filter_size=5, stddev=0.02, padding='SAME', name='deconv2d', reuse=False):
    """
    :param x:
    :param output_size:
    :param sn: use spectral_norm or not
    :param stride:
    :param filter_size:
    :param stddev:
    :param padding:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        shape = x.shape.as_list()
        filter_shape = [filter_size, filter_size, output_size[-1], shape[-1]]
        strides_shape = [1, stride, stride, 1]
        w = tf.get_variable('w', filter_shape, tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        if sn:
            w = spectral_norm(w)
        b = tf.get_variable('b', [output_size[-1]], tf.float32, tf.constant_initializer(0.0))
        return tf.nn.bias_add(tf.nn.conv2d_transpose(x, filter=w, output_shape=output_size,
                                                     strides=strides_shape, padding=padding), b)


def res_block(x, name='res_block', sn=True, reuse=False):
    """
    keep size, 1 * 1 + 3 * 3 + 1 * 1
    :param x:
    :param name:
    :param sn: use spectral_norm or not
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        batch_, h_, w_, output_num = x.shape.as_list()
        x1 = conv2d(x, output_num=output_num / 2, sn=sn, stride=1, filter_size=1, padding='VALID', name='conv1')  # 1 * 1 conv
        x1 = lrelu(instance_normalizer(x1, name='bn1'))
        x2 = conv2d(x1, output_num=output_num / 2, sn=sn, stride=1, filter_size=3, padding='SAME', name='conv2')  # 3 * 3 conv
        x2 = lrelu(instance_normalizer(x2, name='bn2'))
        x3 = conv2d(x2, output_num=output_num, sn=sn, stride=1, filter_size=1, padding='VALID', name='conv3')  # 1 * 1 conv
        x3 = instance_normalizer(x3, name='bn3')
        return lrelu(x3 + x)


def res_block3_3(x, name='res_block3_3', sn=True, reuse=False):
    """
    keep size, 3 * 3 + 3 * 3
    :param x:
    :param name:
    :param sn: use spectral_norm or not
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        batch_, h_, w_, output_num = x.shape.as_list()
        x1 = conv2d(x, output_num=output_num, sn=sn, stride=1, filter_size=3, padding='SAME', name='conv1')  # 3 * 3 conv
        x1 = lrelu(instance_normalizer(x1, name='bn1'))
        x2 = conv2d(x1, output_num=output_num, sn=sn, stride=1, filter_size=3, padding='SAME', name='conv2')  # 3 * 3 conv
        x2 = instance_normalizer(x2, name='bn3')
        return lrelu(x2 + x)


def lrelu(x, leak=0.2, name='lrelu'):
    """
    :param x:
    :param leak:
    :param name:
    :return:
    """
    return tf.maximum(x, leak * x, name=name)


def resize_nn(x, resize_h, resize_w):
    """
    :param x:
    :param resize_h: the result image height
    :param resize_w: the result image width
    :return: resized image
    """
    return tf.image.resize_nearest_neighbor(x, size=(int(resize_h), int(resize_w)))


def spectral_norm(w, iteration=1):
    """
    :param w:
    :param iteration:
    :return:
    """
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def l2_norm(v, eps=1e-12):
    """
    :param v:
    :param eps: escape from / 0
    :return:
    """
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def routing(input_u, routing_b, r_iter=3):
    """
    :param input_u:
    :param routing_b:
    :param r_iter:
    :return:
    """
    # [batch_size, 1152, 10, 16, 1]
    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    b, ni, nj, v, _ = input_u.shape.as_list()
    assert input_u.shape == [b, 1152, 10, 16, 1]
    input_u_stopped = tf.stop_gradient(input_u, name='stop_gradient')
    for ite in range(r_iter):
        # [batch_size, 1152, 10, 1, 1]
        routing_c = tf.nn.softmax(routing_b, axis=1)
        assert routing_c.shape == [b, 1152, 10, 1, 1]
        if ite == r_iter - 1:
            # [batch_size, 1, 10, 16, 1]
            routing_s = tf.reduce_sum(tf.multiply(routing_c, input_u), axis=1)
            routing_s = tf.reshape(routing_s, shape=[b, 1, nj, v, 1])
            assert routing_s.shape == [b, 1, 10, 16, 1]
            # [batch_size, 1, 10, 16, 1]
            routing_v = squash(routing_s)
            assert routing_v.shape == [b, 1, 10, 16, 1]
        else:
            # [batch_size, 1, 10, 16, 1]
            routing_s = tf.reduce_sum(tf.multiply(routing_c, input_u_stopped), axis=1)
            routing_s = tf.reshape(routing_s, shape=[b, 1, nj, v, 1])
            assert routing_s.shape == [b, 1, 10, 16, 1]
            # [batch_size, 1, 10, 16, 1]
            routing_v = squash(routing_s)
            assert routing_v.shape == [b, 1, 10, 16, 1]
            # [batch_size, 1152, 10, 16, 1]
            routing_v_tile = tf.tile(routing_v, [1, ni, 1, 1, 1])
            assert routing_v_tile.shape == [b, 1152, 10, 16, 1]
            # [batch_size, 1152, 10, 1, 1]
            routing_b = routing_b + tf.matmul(tf.transpose(input_u_stopped, perm=[0, 1, 2, 4, 3]), routing_v_tile)
            assert routing_b.shape == [b, 1152, 10, 1, 1]
    return routing_v


def squash(vector):
    """
    :param vector:
    :return:
    """
    # vector [batch_size, output_num, vec_len, 1] or [batch_size, 1, output_num, vec_len, 1]
    epsilon = 1e-9
    vector_norm_square = tf.reduce_sum(tf.square(vector), axis=-2, keep_dims=True)
    vector_norm = tf.sqrt(vector_norm_square)
    vector_s = (vector_norm_square / (1 + vector_norm_square) / (vector_norm + epsilon)) * vector
    return vector_s

