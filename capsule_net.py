# -*-coding:utf-8-*-
# https://github.com/czzyyy/CapsNet-Tensorflow
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import Capsule.ops as ops
import Capsule.utils as utl
import Capsule.capsule_block as capsule_block

from tensorflow.examples.tutorials.mnist import input_data
import math


class CapsuleNet(object):
    def __init__(self, batch_size, input_size, class_num, training_epochs, check_step, m_plus=0.9, m_minus=0.1,
                 lamda=0.5, learning_rate=0.01):
        self.batch_size = batch_size
        self.input_size = input_size
        self.class_num = class_num
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.check_step = check_step
        self.epsilon = 1e-9
        self.train_data = None
        self.chunk_size = None
        self.test_data = None

    def _load_mnist_train(self):
        mnist = input_data.read_data_sets("/home/ziyangcheng/datasets/mnist/", one_hot=True)
        self.train_data = mnist.train
        self.chunk_size = int(math.ceil(float(len(mnist.train.images)) / float(self.batch_size)))
        print('Chunk_size(mnist):', self.chunk_size)
        print('Full dataset tensor(mnist):', self.train_data.images.shape)

    def _load_mnist_test(self):
        mnist = input_data.read_data_sets("/home/ziyangcheng/datasets/mnist/", one_hot=True)
        self.test_data = mnist.test
        print('Test size:', self.test_data.images.shape)

    def _build_capsule_net(self, x):
        with tf.variable_scope('conv_layer'):
            x1 = ops.conv2d(x, output_num=256, sn=False, stride=1, filter_size=9, padding='VALID')
            # [batch_size, 20, 20, 256]
            x1 = tf.nn.relu(x1)
            assert x1.shape == [self.batch_size, 20, 20, 256]
        with tf.variable_scope('primary_caps'):
            b, h, w, c = x1.shape.as_list()
            # [b, h, w, v, c]
            x1 = tf.reshape(x1, shape=[b, h, w, 1, c])
            assert x1.shape == [self.batch_size, 20, 20, 1, 256]
            # [batch_size, 1152, 8, 1]
            x2 = capsule_block.conv_capsule(x1)
            assert x2.shape == [self.batch_size, 1152, 8, 1]
        with tf.variable_scope('digit_caps'):
            # [batch_size, 10, 16, 1]
            x3 = capsule_block.fc_capsule(x2)
            assert x3.shape == [self.batch_size, 10, 16, 1]
            return x3

    def _build_reconstruct_net(self, x_masked):
        with tf.variable_scope('reconstruct_net'):
            x_masked_flap = tf.reshape(x_masked, shape=[self.batch_size, -1])
            h0 = ops.full_connect(x_masked_flap, output_num=512, sn=False, name='r_full0')
            h0 = tf.nn.elu(h0)
            h1 = ops.full_connect(h0, output_num=1024, sn=False, name='r_full1')
            h1 = tf.nn.elu(h1)
            h2 = ops.full_connect(h1, output_num=784, sn=False, name='r_full2')
            r_x = tf.nn.elu(h2)
            return r_x

    def test_net(self, batch_num):
        self._load_mnist_test()
        # [batch_size, 28, 28]
        test_x = tf.placeholder(shape=[self.batch_size] + self.input_size, dtype=tf.float32, name='test_x')
        # [batch_size, 10]
        test_y = tf.placeholder(shape=[self.batch_size, self.class_num], dtype=tf.float32, name='test_y')
        # [batch_size, 10, 16, 1]
        capsule_output = self._build_capsule_net(test_x)
        # [batch_size, 10, 16]
        capsule_masked = tf.multiply(tf.squeeze(capsule_output), tf.reshape(test_y, (-1, self.class_num, 1)))
        with tf.variable_scope('reconstruct'):
            input_x_r = self._build_reconstruct_net(capsule_masked)
            assert input_x_r.shape == [self.batch_size, 784]
        # [batch_size, 10, 1, 1]
        capsule_length = tf.sqrt(tf.reduce_sum(tf.square(capsule_output), axis=2, keep_dims=True) + self.epsilon)
        assert capsule_length.shape == [self.batch_size, 10, 1, 1]
        # [batch_size, 10, 1, 1]
        capsule_softmax = tf.nn.softmax(capsule_length, axis=1)
        assert capsule_softmax.shape == [self.batch_size, 10, 1, 1]
        # b). pick out the index of max softmax val of the 10 caps
        # [batch_size, 10, 1, 1] => [batch_size] (index)
        argmax_idx = tf.to_int32(tf.argmax(capsule_softmax, axis=1))
        argmax_idx = tf.reshape(argmax_idx, shape=(self.batch_size, 1))
        argmax_input_y = tf.to_int32(tf.argmax(test_y, axis=1))
        argmax_input_y = tf.reshape(argmax_input_y, shape=(self.batch_size, 1))
        accuracy = tf.reduce_sum(tf.cast(tf.equal(argmax_idx, argmax_input_y), tf.float32)) / self.batch_size
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, '/home/ziyangcheng/python_save_file/capsule/save_model/capsule.ckpt')
            total_accuracy = 0.0
            for j in range(batch_num):
                batch_xs, batch_ys = self.test_data.next_batch(self.batch_size)
                batch_xs = np.reshape(batch_xs, newshape=[-1, ] + self.input_size)
                test_accuracy, test_x_re = sess.run([accuracy, input_x_r], feed_dict={test_x:batch_xs, test_y:batch_ys})
                total_accuracy = total_accuracy + test_accuracy
                for i in range(len(test_x_re)):
                    print('index', j * self.batch_size + i)
                    scipy.misc.imsave(
                        '/home/ziyangcheng/python_save_file/capsule/test/' + str(
                            j * self.batch_size + i) + 'reconstruct.png',
                        np.reshape(test_x_re[i], newshape=[self.input_size[0], self.input_size[1]]))
            total_accuracy = total_accuracy / batch_num
            print('test accuracy: ', total_accuracy)
        print('test done!')

    def test_format_generate(self):
        self._load_mnist_test()
        # [batch_size, 28, 28]
        test_x = tf.placeholder(shape=[self.batch_size] + self.input_size, dtype=tf.float32, name='test_x')
        # [batch_size, 10]
        test_y = tf.placeholder(shape=[self.batch_size, self.class_num], dtype=tf.float32, name='test_y')
        mask = tf.placeholder(shape=[self.batch_size, 10, 16], dtype=tf.float32, name='mask')

        # [batch_size, 10, 16, 1]
        capsule_output = self._build_capsule_net(test_x)
        # [batch_size, 10, 16]
        capsule_masked = tf.multiply(tf.squeeze(capsule_output), tf.reshape(test_y, (-1, self.class_num, 1)))
        with tf.variable_scope('reconstruct'):
            input_x_r = self._build_reconstruct_net(mask)

        index_tensor = tf.to_int32(tf.argmax(test_y, axis=1))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, '/home/ziyangcheng/python_save_file/capsule/save_model/capsule.ckpt')
            # format 0.05 in the range [−0.25, 0.25]
            batch_xs, batch_ys = self.test_data.next_batch(self.batch_size)
            batch_xs = np.reshape(batch_xs, newshape=[-1, ] + self.input_size)

            # index
            index = sess.run(index_tensor, feed_dict={test_y: batch_ys})

            # mask
            mask_value = sess.run(capsule_masked, feed_dict={test_x: batch_xs, test_y: batch_ys})

            f, a = plt.subplots(16, 10, figsize=(10, 16))
            for j in range(16):
                for i in range(10):
                    mask_value[:, index, j] = -0.25 + 0.05 * i
                    images = sess.run(input_x_r, feed_dict={mask: mask_value})
                    a[j][i].imshow(np.reshape(images[j], (28, 28)), cmap="gray")
            f.savefig('/home/ziyangcheng/python_save_file/capsule/test/format.png')
        print('test done!')

    def train_net(self):
        # load dataset
        self._load_mnist_train()
        with tf.variable_scope('inputs'):
            # [batch_size, 28, 28]
            input_x = tf.placeholder(shape=[self.batch_size] + self.input_size, dtype=tf.float32, name='input_x')
            # [batch_size, 10]
            input_y = tf.placeholder(shape=[self.batch_size, self.class_num], dtype=tf.float32, name='input_y')
        # [batch_size, 10, 16, 1]
        capsule_output = self._build_capsule_net(input_x)
        assert capsule_output.shape == [self.batch_size, 10, 16, 1]
        # [batch_size, 10, 16]
        capsule_masked = tf.multiply(tf.squeeze(capsule_output), tf.reshape(input_y, (-1, self.class_num, 1)))
        assert capsule_masked.shape == [self.batch_size, 10, 16]
        with tf.variable_scope('reconstruct'):
            input_x_r = self._build_reconstruct_net(capsule_masked)
            assert input_x_r.shape == [self.batch_size, 784]
        with tf.variable_scope('loss'):
            # [batch_size, 10, 1, 1]
            capsule_length = tf.sqrt(tf.reduce_sum(tf.square(capsule_output), axis=2, keep_dims=True) + self.epsilon)
            assert capsule_length.shape == [self.batch_size, 10, 1, 1]
            # [batch_size, 10]
            capsule_length_squeeze = tf.squeeze(capsule_length)
            assert capsule_length_squeeze.shape == [self.batch_size, 10]
            # [batch_size, 10]
            loss_comp_one = tf.maximum(0.0, self.m_plus - capsule_length_squeeze)  # element_wise
            assert loss_comp_one.shape == [self.batch_size, 10]
            # [batch_size, 10]
            loss_comp_two = tf.maximum(0.0, capsule_length_squeeze - self.m_minus)
            assert loss_comp_two.shape == [self.batch_size, 10]
            # [batch_size, 10]
            classify_loss = input_y * tf.square(loss_comp_one) + self.lamda * (1 - input_y) * tf.square(loss_comp_two)
            assert classify_loss.shape == [self.batch_size, 10]
            classify_loss = tf.reduce_mean(tf.reduce_sum(classify_loss, axis=1), name='classify_loss')

            input_x_flap = tf.reshape(input_x, shape=(self.batch_size, -1))
            reconstruct_loss = 0.3 * tf.reduce_mean(tf.square(input_x_r - input_x_flap) / 2, name='reconstruct_loss')

            total_loss = classify_loss + reconstruct_loss

            tf.summary.scalar('classify_loss', classify_loss)
            tf.summary.scalar('reconstruct_loss', reconstruct_loss)
            tf.summary.scalar('total_loss', total_loss)

        with tf.variable_scope('accuracy'):
            # [batch_size, 10, 1, 1]
            capsule_softmax = tf.nn.softmax(capsule_length, axis=1)
            assert capsule_softmax.shape == [self.batch_size, 10, 1, 1]

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            argmax_idx = tf.to_int32(tf.argmax(capsule_softmax, axis=1))
            argmax_idx = tf.reshape(argmax_idx, shape=(self.batch_size, 1))
            argmax_input_y = tf.to_int32(tf.argmax(input_y, axis=1))
            argmax_input_y = tf.reshape(argmax_input_y, shape=(self.batch_size, 1))
            accuracy = tf.reduce_sum(tf.cast(tf.equal(argmax_idx, argmax_input_y), tf.float32)) / self.batch_size

            tf.summary.scalar('accuracy', accuracy)

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        with tf.variable_scope('train'):
            with tf.Session() as sess:
                saver = tf.train.Saver()
                # merge summary
                merged = tf.summary.merge_all()
                # choose dir
                writer = tf.summary.FileWriter('/home/ziyangcheng/python_save_file/capsule/tf_board/', sess.graph)
                sess.run(tf.global_variables_initializer())
                for e in range(self.training_epochs):
                    for batch_i in range(self.chunk_size):
                        # mnist
                        batch_xs, batch_ys = self.train_data.next_batch(self.batch_size)
                        batch_data = np.reshape(batch_xs, newshape=[-1, ] + self.input_size)

                        # Run optimizers
                        sess.run(optimizer, feed_dict={input_x: batch_data, input_y: batch_ys})

                        if (self.chunk_size * e + batch_i) % self.check_step == 0:
                            train_classify_loss = sess.run(classify_loss, feed_dict={input_x: batch_data, input_y: batch_ys})
                            train_reconstruct_loss = sess.run(reconstruct_loss, feed_dict={input_x: batch_data, input_y: batch_ys})
                            train_total_loss = sess.run(total_loss, feed_dict={input_x: batch_data, input_y: batch_ys})
                            train_accuracy = sess.run(accuracy, feed_dict={input_x: batch_data, input_y:batch_ys})
                            check_image = sess.run(input_x_r, feed_dict={input_x: batch_data, input_y: batch_ys})

                            merge_result = sess.run(merged, feed_dict={input_x: batch_data, input_y: batch_ys})

                            writer.add_summary(merge_result, self.chunk_size * e + batch_i)

                            print(
                                "step {}/of epoch {}/{}...".format(self.chunk_size * e + batch_i, e,
                                                                   self.training_epochs),
                                "Classify Loss: {:.4f}".format(train_classify_loss),
                                "Reconstruct Loss: {:.4f}".format(train_reconstruct_loss),
                                "Total Loss: {:.4f}".format(train_total_loss),
                                "Accuracy: {:.4f}".format(train_accuracy))

                            # show pic
                            scipy.misc.imsave('/home/ziyangcheng/python_save_file/capsule/train/' + str(
                                self.chunk_size * e + batch_i) +
                                              '-' + str(0) + '.png', np.reshape(check_image[0],
                                                                                newshape=[self.input_size[0],
                                                                                          self.input_size[1]]))

                print('train done')
                # save sess
                saver.save(sess, '/home/ziyangcheng/python_save_file/capsule/save_model/capsule.ckpt')

