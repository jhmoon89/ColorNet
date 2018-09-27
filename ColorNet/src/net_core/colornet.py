import tensorflow as tf
import numpy as np
import cv2
import time
import os
import sys

class ColorNet_core(object):
    def __init__(self, name='colornet_core', trainable=True, bnPhase=True, reuse=False, activation=tf.nn.elu):
        self._reuse = reuse
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._activation = activation
        self._name = name
        self.variables = None
        self.update_ops = None
        self.saver = None

        # print('init func')

    def _conv(self, inputs, filters, kernel_size, strides=1, dilations=1, batch_norm_flag=False):
        # print inputs.get_shape()
        hidden = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            dilation_rate=dilations, activation=None, trainable=self._trainable, use_bias=False,
            reuse=False
        )
        if batch_norm_flag:
            hidden = tf.layers.batch_normalization(hidden, training=self._bnPhase, trainable=self._trainable)
        hidden = self._activation(hidden)
        # print hidden.get_shape()
        return hidden

    def _conv_trans(self, inputs, filters, kernel_size, strides=1, batch_norm_flag=False):
        hidden = tf.layers.conv2d_transpose(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            activation=None, trainable=self._trainable, use_bias=False,
            reuse=False
        )
        if batch_norm_flag:
            hidden = tf.layers.batch_normalization(hidden, training=self._bnPhase, trainable=self._trainable)
        hidden = self._activation(hidden)
        # print hidden.get_shape()
        return hidden

    def _maxpool(self, inputs, pool_size=(2, 2), strides=2, padding='same'):
        hidden = tf.layers.max_pooling2d(inputs=inputs, pool_size=pool_size, strides=strides, padding=padding)
        return hidden

    def __call__(self, InputImgs):
        # print self._nameScope

        with tf.variable_scope(self._name, reuse=self._reuse):
            h11 = self._conv(inputs=InputImgs, filters=64, kernel_size=1)
            h12 = self._conv(inputs=h11, filters=64, kernel_size=1, strides=2, batch_norm_flag=True)

            h21 = self._conv(inputs=h12, filters=128, kernel_size=1)
            h22 = self._conv(inputs=h21, filters=128, kernel_size=1, strides=2, batch_norm_flag=True)

            h31 = self._conv(inputs=h22, filters=256, kernel_size=1)
            h32 = self._conv(inputs=h31, filters=256, kernel_size=1)
            h33 = self._conv(inputs=h32, filters=256, kernel_size=1, strides=2, batch_norm_flag=True)

            h41 = self._conv(inputs=h33, filters=512, kernel_size=1)
            h42 = self._conv(inputs=h41, filters=512, kernel_size=1)
            h43 = self._conv(inputs=h42, filters=512, kernel_size=1, batch_norm_flag=True)

            h51 = self._conv(inputs=h43, filters=512, kernel_size=1, strides=1, dilations=2)
            h52 = self._conv(inputs=h51, filters=512, kernel_size=1, strides=1, dilations=2)
            h53 = self._conv(inputs=h52, filters=512, kernel_size=1, strides=1, dilations=2, batch_norm_flag=True)

            h61 = self._conv(inputs=h53, filters=512, kernel_size=1, strides=1, dilations=2)
            h62 = self._conv(inputs=h61, filters=512, kernel_size=1, strides=1, dilations=2)
            h63 = self._conv(inputs=h62, filters=512, kernel_size=1, strides=1, dilations=2, batch_norm_flag=True)

            h71 = self._conv(inputs=h63, filters=256, kernel_size=1)
            h72 = self._conv(inputs=h71, filters=256, kernel_size=1)
            h73 = self._conv(inputs=h72, filters=256, kernel_size=1, batch_norm_flag=True)

            h81 = self._conv_trans(inputs=h73, filters=128, kernel_size=1, strides=2)
            h82 = self._conv(inputs=h81, filters=128, kernel_size=1)
            h83 = self._conv(inputs=h82, filters=128, kernel_size=1, batch_norm_flag=True)

        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._name)
        self.saver = tf.train.Saver(var_list=self.variables)
        outputs = h83

        return outputs

class CN_Colorize(object):
    def __init__(self, outputDim, name='cn', trainable=True,
                 bnPhase=True, reuse=False, coreActivation=tf.nn.leaky_relu,
                 lastLayerActivation=None,
                 lastLayerPooling=None):
        self._outputDim = outputDim
        self._name = name
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._reuse = reuse
        self._coreActivation = coreActivation
        self._lastActivation = lastLayerActivation
        self._lastPool = lastLayerPooling
        self.variables = None
        self.update_ops = None
        self.saver = None
        self._CNet_core = None

        # print 'init'

    def __call__(self, InputImgs):
        # print self._nameScope
        self._CNet_core = ColorNet_core(name=self._name+"_CNCore", trainable=self._trainable,
                                        bnPhase=self._bnPhase, reuse=self._reuse, activation=self._coreActivation)

        hidden = self._CNet_core(InputImgs)

        with tf.variable_scope(self._name+'_Detection', reuse=self._reuse):
            # h1 = self._CNet_core._conv_trans(inputs=hidden, filters=128, kernel_size=1, strides=2)
            # h2 = self._CNet_core._conv_trans(inputs=h1, filters=128, kernel_size=1, strides=2)
            # output = tf.layers.conv2d(inputs=h2, filters=self._outputDim, kernel_size=3, strides=1, padding='same',
            #                           activation=None, trainable=self._trainable, use_bias=False)
            # print 'output shape is {}'.format(output.shape)
            # h1 = tf.layers.conv2d(inputs=hidden, filters=128, kernel_size=3, padding='same')
            # h2 = tf.layers.conv2d(inputs=h1, filters=128, kernel_size=3, padding='same')
            output = tf.layers.conv2d(inputs=hidden, filters=self._outputDim, kernel_size=3, padding='same')

        self._reuse = True
        self.variables = [self._CNet_core.variables,
                          tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name+"_Detection")]
        self.update_ops = [self._CNet_core.update_ops,
                           tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._name+"_Detection")]
        self.allVariables = self._CNet_core.variables + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                          scope=self._name+"_Detection")
        self.allUpdate_ops = self._CNet_core.update_ops + tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                                            scope=self._name+"_Detection")
        self.coreVariables = self._CNet_core.variables
        self.colorizorVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name+"_Detection")
        self.coreSaver = tf.train.Saver(var_list=self.coreVariables,
                                        max_to_keep=4, keep_checkpoint_every_n_hours=2)
        self.colorizorSaver = tf.train.Saver(var_list=self.colorizorVariables,
                                             max_to_keep=4, keep_checkpoint_every_n_hours=2)

        return output


# x = tf.placeholder(tf.float32, [None, 224, 224, 4], 'input')
# y = CN_Colorize(374)
# out = y(x)
# print out.get_shape()
