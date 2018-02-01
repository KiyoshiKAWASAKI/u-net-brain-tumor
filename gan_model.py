# Please be noticed that unet code is borrowed from: 
# https://github.com/zsdonghao/u-net-brain-tumor

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow import *
import tensorlayer as tl
from tensorlayer.layers import *

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

# Define discriminator (use auto encoder)
def discriminator(inputs, is_train=True, reuse=False):
	n_filter = 32
	_, nx, ny, nz = inputs.get_shape().as_list()
	w_init = tf.random_normal_initializer(stddev=0.02)

	with tf.variable_scope("discriminator", reuse=reuse):
		tl.layers.set_name_reuse(reuse)
		d_input = InputLayer(inputs, name='d_input') # [10, 240, 240, 2]

		# Encoder
		conv0 = Conv2d(d_input, n_filter, 
						(3, 3), (2, 2), 
						act=tf.nn.elu,
                        padding='SAME', 
                        W_init=w_init, 
                        name='conv0') # (nx/2,ny/2,n)

		conv1 = Conv2d(net_conv0, 2*n_filter, 
						(3, 3), (2, 2), 
						act=tf.nn.elu,
                        padding='SAME', 
                        W_init=w_init, 
                        name='conv1') # (nx/4,ny/4,2*n)
		
		conv2 = Conv2d(net_conv1, 4*n_filter, 
						(3, 3), (2, 2), 
						act=tf.nn.elu,
                        padding='SAME', 
                        W_init=w_init, 
                        name='conv2') # (nx/8,ny/8,4*n)

        # Decoder
        deconv2 = DeConv2d(conv2, 2*n_filter, (3, 3), (nx/4, ny/4), (2, 2), name='deconv2')
        deconv1 = DeConv2d(net_deconv2, n_filter, (3, 3), (nx/2, ny/2), (2, 2), name='deconv1')
        deconv0 = DeConv2d(net_deconv1, nz, (3, 3), (nx, ny), (2, 2), act=tf.nn.sigmoid, name='deconv0')

	return deconv0.outputs

"""
def discriminator(x, is_train=True, reuse=False):
	with tf.variable_scope("discriminator", reuse=reuse):
		tl.layers.set_name_reuse(reuse)
		nx = ny = 240

		# Input size: [10,240,240,1]
		inputs = x
		inputs = inputs + tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=0.01)
		inputs = tf.reshape(inputs, (-1, 240, 240, 1))
		inputs = InputLayer(inputs, name='inputs')

		conv1 = Conv2d(inputs, 32, (4, 4), act=tf.nn.elu, name='conv1')
		conv2 = Conv2d(conv1, 64, (4, 4), act=tf.nn.elu, name='conv2')
		ffc2 = Conv2d(conv2, 128, (4, 4), act=tf.nn.elu, name='ffc2')
		
		ffc1 = FlattenLayer(ffc2, name='flatten_layer')
		ffc1 = DenseLayer(ffc1, 512, act=tf.nn.elu, name='ffc1')
		zz = DenseLayer(ffc1, 200, act=tf.nn.elu, name='zz')
		fc1 = DenseLayer(zz, 512, act=tf.nn.elu, name='fc1')
		fc2 = DenseLayer(fc1, 128*8*8, act=tf.nn.elu, name='fc2')
		fc2 = tf.reshape(fc2.outputs, (-1, 8, 8, 128))
		fc2_input = InputLayer(fc2, name='inputs_fc2')
		
		deconv1 = DeConv2d(ffc2, 64, (4, 4), (nx/4, ny/4), (2, 2), name='deconv1')
		deconv2 = DeConv2d(deconv1, 32, (4, 4), (nx/2, ny/2), (2, 2), name='deconv2')
		result = DeConv2d(deconv2, 1, (4, 4), (nx/1, ny/1),(2, 2), act=tf.nn.sigmoid, name='output')
	
	resid = tf.abs(x - result.outputs)
	loss=tf.reduce_mean(resid)

	return loss
"""

