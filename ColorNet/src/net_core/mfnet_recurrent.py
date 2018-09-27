import tensorflow as tf
import numpy as np
import cv2
import time
import os
import sys

class mfnet_core(object):
    def __init__(self, timestep=4, name='mfnet_core', trainable=True,
                 bnPhase=True, reuse=tf.AUTO_REUSE, activation = tf.nn.elu):
        # self._reuse = reuse
        self._reuse = reuse
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._activation = activation
        self._name = name
        self._timestep = timestep
        self.variables = None
        self.update_ops = None
        self.saver = None

        # print('init func')

    def _conv(self, inputs, filters, kernel_size, strides=1):
        hidden = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding='same', activation=None, trainable=self._trainable, use_bias=False,
        )
        hidden = tf.layers.batch_normalization(hidden, training=self._bnPhase, trainable=self._trainable)
        hidden = self._activation(hidden)
        return hidden

    def _maxpool(self, inputs, pool_size=(2,2), strides=2, padding='same'):
        hidden = tf.layers.max_pooling2d(inputs=inputs, pool_size=pool_size, strides=strides, padding=padding)
        return hidden

    def __call__(self, InputImgs):
        # print self._nameScope

        h_list = []
        for t in range(self._timestep):
            with tf.variable_scope(self._name, reuse=self._reuse) as mf_conv:
                h1 = self._conv(inputs=InputImgs[:, t, ...], filters=32, kernel_size=3)
                p1 = self._maxpool(inputs=h1)

                h2 = self._conv(inputs=p1, filters=64, kernel_size=3)
                p2 = self._maxpool(inputs=h2)

                h31 = self._conv(inputs=p2, filters=128, kernel_size=3)
                h32 = self._conv(inputs=h31, filters=64, kernel_size=1)
                h33 = self._conv(inputs=h32, filters=128, kernel_size=3)
                p3 = self._maxpool(inputs=h33)

                h41 = self._conv(inputs=p3, filters=256, kernel_size=3)
                h42 = self._conv(inputs=h41, filters=128, kernel_size=1)
                h43 = self._conv(inputs=h42, filters=256, kernel_size=3)
                p4 = self._maxpool(inputs=h43)

                h51 = self._conv(inputs=p4, filters=512, kernel_size=3)
                h52 = self._conv(inputs=h51, filters=256, kernel_size=1)
                h53 = self._conv(inputs=h52, filters=512, kernel_size=3)
                h54 = self._conv(inputs=h53, filters=256, kernel_size=1)
                h55 = self._conv(inputs=h54, filters=512, kernel_size=3)
                p5 = self._maxpool(inputs=h55)

                h61 = self._conv(inputs=p5, filters=1024, kernel_size=3)
                h62 = self._conv(inputs=h61, filters=512, kernel_size=1)
                h63 = self._conv(inputs=h62, filters=1024, kernel_size=3)
                h64 = self._conv(inputs=h63, filters=512, kernel_size=1)
                h65 = self._conv(inputs=h64, filters=1024, kernel_size=3)

            # print h65.name
            h_list.append(h65)

        # self.varScope = mf_conv
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._name)
        self.saver = tf.train.Saver(var_list=self.variables)
        outputs = tf.stack(h_list, 1)
        # print outputs.get_shape()

        return outputs


# class MF_Classification(object):
#     def __init__(self, outputDim, nameScope='mf_rpn', trainable=True,
#                  bnPhase=True, reuse=False, coreActivation=tf.nn.leaky_relu,
#                  lastLayerActivation=None,
#                  lastLayerPooling=None):
#         self._outputDim = outputDim
#         self._nameScope = nameScope
#         self._trainable = trainable
#         self._bnPhase = bnPhase
#         self._reuse = reuse
#         self._coreActivation = coreActivation
#         self._lastActivation = lastLayerActivation
#         self._lastPool = lastLayerPooling
#         self.variables = None
#         self.update_ops = None
#         self.saver = None
#         self._mfnet_core = None
#
#     def __call__(self, InputImgs):
#         # print self._nameScope
#         self._mfnet_core = mfnet_core(nameScope=self._nameScope+"_MFNetCore", trainable=self._trainable,
#                                       bnPhase=self._bnPhase, reuse=self._reuse, activation=self._coreActivation)

class mfnet_last(object):
    def __init__(self, outputDim, timestep=4, name='mfnet_last',
                 trainable=True, bnPhase=True, reuse=tf.AUTO_REUSE, activation = None):
        # self._reuse = reuse
        self._reuse = reuse
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._activation = activation
        self._name = name
        self._timeStep = timestep
        self.variables = None
        self.update_ops = None
        self.saver = None
        self._outputDim = outputDim

        # print('init func')

    def _conv3d(self, inputs, filters, kernel_size, strides=1):
        hidden = tf.layers.conv3d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding='valid', activation=None, trainable=self._trainable, use_bias=False
        )
        hidden = tf.layers.batch_normalization(hidden, training=self._bnPhase, trainable=self._trainable)
        if self._activation == None:
            return hidden
        else:
            hidden = self._activation(hidden)
            return hidden

    def _conv2d(self, inputs, filters, kernel_size, strides=1):
        hidden = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding='same', activation=None, trainable=self._trainable, use_bias=False
        )
        hidden = tf.layers.batch_normalization(hidden, training=self._bnPhase, trainable=self._trainable)
        if self._activation == None:
            return hidden
        else:
            hidden = self._activation(hidden)
            return hidden

    def _maxpool(self, inputs, pool_size=(2,2), strides=2, padding='same'):
        hidden = tf.layers.max_pooling2d(inputs=inputs, pool_size=pool_size, strides=strides, padding=padding)
        return hidden

    def __call__(self, hidden):
        # print self._nameScope

        with tf.variable_scope(self._name, reuse=self._reuse) as mf_last:
            last_hidden = self._conv3d(inputs=hidden, filters=1024, kernel_size=(3, 1, 1))
            output = self._conv3d(inputs=last_hidden, filters=1024, kernel_size=(1, 1, 1))

            # print 'h3 shape is {}'.format(h3.shape)

        # print output.get_shape()

        # self.varScope = mf_conv
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._name)
        self.saver = tf.train.Saver(var_list=self.variables)

        return output


class MF_Detection(object):
    def __init__(self, outputDim, timestep=4, name='mf_det', trainable=True,
                 bnPhase=True, reuse=tf.AUTO_REUSE, coreActivation=tf.nn.leaky_relu,
                 lastLayerActivation=None,
                 lastLayerPooling=None):
        self._outputDim = outputDim
        self._timeStep = timestep
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
        self._mfnet_core = None
        self._mfnet_last = None

        # print 'init'

    def __call__(self, InputImgs):
        # print self._nameScope
        self._mfnet_core = mfnet_core(timestep=self._timeStep, name=self._name+"_MFCore", trainable=self._trainable,
                                      bnPhase=self._bnPhase, reuse=self._reuse, activation=self._coreActivation)

        # with tf.variable_scope(self._name+"_recurrent") as scope:


        # for t in range(self._timestep):
        hidden = self._mfnet_core(InputImgs)
        # print hidden.get_shape()

        self._mfnet_last = mfnet_last(outputDim=self._outputDim, timestep=self._timeStep, name=self._name+"_MFLast",
                                      trainable=self._trainable, bnPhase=self._bnPhase, reuse=self._reuse,
                                      activation=self._lastActivation
                                      )

        output_flow = self._mfnet_last(hidden)
        # print output_flow.get_shape()

        # # # Now split hidden feature to several parts; cls_feature, current_loc_feature, movement_feature
        # hidden_cls = hidden[:, :, :256, ...]
        # hidden_x = hidden[:, :, 256:768, ...]
        # hidden_next = hidden[:, :, 768:, ...]
        #
        # # hidden_v_first = tf.expand_dims(tf.zeros(tf.shape(hidden_v[..., 0])), -1) ### first v as zero (v0 = 0)
        # hidden_v_first = tf.expand_dims(hidden_next[..., 0], -1) ### first v as v1 (v0 = v1)
        # new_hidden_next = (tf.concat([hidden_v_first, hidden_next], -1))[..., :-1]
        #
        # new_hidden = tf.concat([hidden_cls, hidden_x, new_hidden_next], -2)
        #
        # # print new_hidden.get_shape()
        #
        # self._mfnet_last = mfnet_last(
        #     outputDim=self._outputDim, timestep=self._timeStep
        #     , name=self._name+"_MFLast", trainable=self._trainable
        #     , bnPhase=self._bnPhase, reuse=self._reuse, activation=self._lastActivation
        # )
        #
        # outputs = self._mfnet_last(new_hidden)
        #
        # # print outputs.get_shape()
        #
        out_list = []
        for t in range(self._timeStep):
            with tf.variable_scope(self._name+'_Detection', reuse=self._reuse):
                h1 = tf.layers.conv2d(inputs=hidden[:, t, ...], filters=1024, kernel_size=3, strides=1, padding='same',
                                      activation=None, trainable=self._trainable, use_bias=False)
                h1 = tf.layers.batch_normalization(h1, training=self._bnPhase, trainable=self._trainable)
                h2 = tf.layers.conv2d(inputs=h1, filters=1024, kernel_size=3, strides=1, padding='same',
                                      activation=None, trainable=self._trainable, use_bias=False)
                h2 = tf.layers.batch_normalization(h2, training=self._bnPhase, trainable=self._trainable)
                h3 = tf.layers.conv2d(inputs=h2, filters=1024, kernel_size=3, strides=1, padding='same',
                                      activation=None, trainable=self._trainable, use_bias=False)
                h3 = tf.layers.batch_normalization(h3, training=self._bnPhase, trainable=self._trainable)

                # print 'h3 shape is {}'.format(h3.shape)
                output = tf.layers.conv2d(inputs=h3, filters=self._outputDim, kernel_size=1, strides=1, padding='same',
                                          activation=None, trainable=self._trainable, use_bias=False)
            out_list.append(output)
        outputs = tf.stack(out_list, 1)
            # print 'output shape is {}'.format(output.shape)

        self._reuse = True
        self.variables = [self._mfnet_core.variables, self._mfnet_last.variables,
                          tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name+"_Detection")]
        self.update_ops = [self._mfnet_core.update_ops, self._mfnet_last.update_ops,
                           tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._name + "_Detection")]
        self.allVariables = self._mfnet_core.variables + self._mfnet_last.variables \
                            + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name+"_Detection")
        self.allUpdate_ops = self._mfnet_core.update_ops + self._mfnet_last.update_ops \
                            + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._name+"_Detection")
        self.coreVariables = self._mfnet_core.variables
        self.lastVariables = self._mfnet_last.variables
        self.detectVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name+"_Detection")
        # self.secondVariables = self._mfnet_core.secondVar
        self.coreSaver = tf.train.Saver(var_list=self.coreVariables, max_to_keep=4, keep_checkpoint_every_n_hours=2)
        self.lastSaver = tf.train.Saver(var_list=self.lastVariables, max_to_keep=4, keep_checkpoint_every_n_hours=2)
        self.detectSaver = tf.train.Saver(var_list=self.detectVariables, max_to_keep=4, keep_checkpoint_every_n_hours=2)

        return output_flow, outputs


# m1 = MF_Detection(41)
# x = tf.placeholder(tf.float32, [None, 4, 416, 416, 3])
#
# y1, y2 = m1(x)
# print y1.get_shape()
# print y2.get_shape()

# m1 = mfnet_core()
# x = tf.placeholder(tf.float32, [None, 4, 416, 416, 3])
# h = m1(x)
#
# m2 = mfnet_last(41)
# h2 = m2(h)
