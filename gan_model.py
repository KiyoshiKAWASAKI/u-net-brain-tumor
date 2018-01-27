from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import ops
import tensorflow as tf
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

# Define discriminator
def discriminator(self, x, is_train=True, reuse=False):
	with tf.variable_scope("Discriminator", reuse=reuse):
		tl.layers.set_name_reuse(reuse)

		with arg_scope([layers.conv2d],
                      kernel_size=[4, 4],
                      stride=[2, 2],
                      activation_fn=ops.leakyrelu,
                      normalizer_fn=layers.batch_norm,
                      normalizer_params=batch_norm_params):

			x = x + tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=0.01)
			x = tf.reshape(x, (-1, 64, 64, 3))
			
			self.conv1 = layers.conv2d(inputs=x,
                                   		num_outputs=32,
                                   		normalizer_fn=None,
                                   		biases_initializer=None,
                                   		scope='conv1')

			self.conv2 = layers.conv2d(inputs=self.conv1,
                                   num_outputs=64,
                                   scope='conv2')

			self.ffc2 = layers.conv2d(inputs=self.conv2,
                                   num_outputs=128,
                                   scope='ffc2')

			self.ffc1 = layers.fully_connected(
								inputs = self.ffc2,
								num_outputs = 512,
								scope = 'ffc1')

			self.zz = layers.fully_connected(
								inputs = self.ffc1,
								num_outputs = 200,
								scope = 'zz')

			self.fc1 = layers.fully_connected(
								inputs = self.zz,
								num_outputs = 512,
								scope = 'ffc1')

			self.fc2 = layers.fully_connected(
								inputs = self.fc1,
								num_outputs = 128*8*8,
								scope = 'ffc1')

			self.fc2 = tf.reshape(self.fc2, (-1, 8, 8, 128))

			self.deconv1 = layers.conv2d_transpose(inputs=self.fc2,
                                              num_outputs=64,
                                              scope='deconv1')

			self.deconv2 = layers.conv2d_transpose(inputs=self.deconv1,
                                              num_outputs=32,
                                              scope='deconv2')

			self.output = layers.conv2d_transpose(inputs=self.deconv2,
                                              num_outputs=3,
                                              scope='deconv1')

			self.out = tf.nn.sigmoid(self.output)
	
	resid = tf.abs(x - self.out)
    loss = tf.reduce_mean(resid)			

	return loss


	"""
    reuse = len([t for t in tf.global_variables() if t.name.startswith('dis_2d')]) > 0
    with tf.sg_context(name='dis_2d', stride=2, act='leaky_relu', bn=bn, reuse=reuse):
        x=x+tf.random_normal(shape=tf.shape(x),mean=0.0,stddev=0.01)
        x=x.sg_reshape(shape=(-1,64,64,3))
        conv1=x.sg_conv(dim=32, size=4, name='gen6')
        conv2=conv1.sg_conv(dim=64,size=4,name='gen7')      
        ffc2=conv2.sg_conv(dim=128,size=4,name='gen8')
        ffc1=ffc2.sg_flatten().sg_dense(dim=512,name='gen9')
        zz=ffc1.sg_dense(dim=200,name='gen10')  
        fc1 = zz.sg_dense(dim=512,name='gen1')
        fc2 = fc1.sg_dense(dim=128*8*8,name='reshape').sg_reshape(shape=(-1, 8, 8, 128), name='gen2')
        deconv1 = fc2.sg_upconv(dim=64, stride=2, size=4, name='gen3')
        deconv2 = deconv1.sg_upconv(dim=32, stride=2, size=4, name='gen4')
        out = deconv2.sg_upconv(dim=3, stride=2, size=4, act='sigmoid', bn=False, name='gen5') 
    resid=tf.abs(x-out)
    loss=tf.reduce_mean(resid)
    return loss
    """

