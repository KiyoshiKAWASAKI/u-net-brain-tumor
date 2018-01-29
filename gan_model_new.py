from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow import *
import tensorlayer as tl
from tensorlayer.layers import *

slim = tf.contrib.slim
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope
FLAGS = tf.app.flags.FLAGS

# Use unet as generator
def u_net(x, is_train=False, reuse=False, n_out=1):
	_, nx, ny, nz = x.get_shape().as_list()
	with tf.variable_scope("u_net", reuse=reuse):
	    tl.layers.set_name_reuse(reuse)
	    inputs = InputLayer(x, name='inputs')
	    conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
	    conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
	    pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
	    conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
	    conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
	    pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
	    conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')
	    conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')
	    pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
	    conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')
	    conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')
	    pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
	    conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')
	    conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')

	    up4 = DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
	    up4 = ConcatLayer([up4, conv4], 3, name='concat4')
	    conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')
	    conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')
	    up3 = DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
	    up3 = ConcatLayer([up3, conv3], 3, name='concat3')
	    conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')
	    conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')
	    up2 = DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
	    up2 = ConcatLayer([up2, conv2], 3, name='concat2')
	    conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu,  name='uconv2_1')
	    conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')
	    up1 = DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
	    up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
	    conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')
	    conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')
	    conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
	return conv1

def leaky_rectify(x, leakiness=0.01):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    return ret


def custom_conv2d(input_layer, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, in_dim=None,
                 padding='SAME', scope="conv2d"):
        with tf.variable_scope(scope):
            w = tf.get_variable('w', [k_h, k_w, in_dim or input_layer.shape[-1], output_dim],
                              initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_layer, w,
                                strides=[1, d_h, d_w, 1], padding=padding)
            b = tf.get_variable("b", shape=output_dim, initializer=tf.constant_initializer(0.))
            conv = tf.nn.bias_add(conv, b)
            return conv



def custom_fc(input_layer, output_size, scope='Linear',
                 in_dim=None, stddev=0.02, bias_start=0.0):
        shape = input_layer.shape
        if len(shape) > 2:
            input_layer = tf.reshape(input_layer, [-1, int(np.prod(shape[1:]))])
        shape = input_layer.shape
        with tf.variable_scope(scope):
            matrix = tf.get_variable("weight",
                                   [in_dim or shape[1], output_size],
                                   dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
            return tf.nn.bias_add(tf.matmul(input_layer, matrix), bias)



