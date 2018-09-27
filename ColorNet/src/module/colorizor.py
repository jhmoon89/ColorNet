import numpy as np
import tensorflow as tf
import cv2
import time
import os
import sys

########################################################
# import inspect

# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

#import net_core.colornet as colornet
########################################################
import src.net_core.colornet as colornet
import dataset_utils.datasetUtils as datasetUtils


class colornet_colorizor(object):
    def __init__(self,
                 dataPath='./',
                 name='CN',
                 imgSize=(224, 224),
                 batchSize=32,
                 learningRate=0.0001,
                 classNum=31,
                 coreActivation=tf.nn.relu,
                 lastActivation=tf.nn.softmax,
                 consecutiveFrame=2,
                 gamut=np.load("/ssdubuntu/color/gamut_cell16.npy")
                 ):
        self._imgList = None
        self._dataPath = dataPath
        self._name = name
        self._imgSize = imgSize
        self._batchSize = batchSize
        self._lr = learningRate
        self._coreAct = coreActivation
        self._lastAct = lastActivation
        self._classNum = classNum
        self.variables = None
        self.update_ops = None
        self._inputImgs = None
        self._output = None
        self._outputGT = None
        self._optimizer = None
        self._loss = None
        self._consecutiveFrame = consecutiveFrame
        self._gamutGrid = gamut

        # initialize Vars
        self._refineGamut()
        self._buildNetwork()
        self._createLoss()
        self._setOptimizer()
        self._createEvaluation()

        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        self._sess = tf.Session()
        self._sess.run(init)

    def _refineGamut(self):
        self._gamutList = np.argwhere(self._gamutGrid != 0)
        self._BinNum = len(self._gamutList)
        gamut_normalize = self._gamutGrid / np.sum(self._gamutGrid)
        ind_mask = np.where(gamut_normalize != 0)
        gamut_blur = cv2.GaussianBlur(gamut_normalize, (5, 5), 5)
        gamut_smooth = np.zeros_like(self._gamutGrid)
        gamut_smooth[ind_mask] = 1.0 / (0.5 * (gamut_blur[ind_mask] + 1 / float(self._BinNum)))
        gamut_smooth = gamut_smooth / np.sum(gamut_smooth)
        self._weight_list = gamut_smooth[ind_mask]
        # print np.shape(self._weight_list)
        # np.save("/ssdubuntu/color/rebalance_weight.npy", weight_list)
        # print self._BinNum

    def _buildNetwork(self):
        print "build Network..."
        #######################################################
        # tf.reset_default_graph()  ### modified for ipython!!!##
        #######################################################
        self._inputImgs = tf.placeholder(tf.float32, shape=[None,
                                                            self._imgSize[0],
                                                            self._imgSize[1],
                                                            1
                                                            ])
        self._outputGT = tf.placeholder(tf.float32, shape=[None,
                                                           self._imgSize[0] / 4,
                                                           self._imgSize[1] / 4,
                                                           self._BinNum
                                                           ])
        # self._outputGT = tf.placeholder(tf.float32, shape=[None,
        #                                                    self._imgSize[0],
        #                                                    self._imgSize[1],
        #                                                    self._BinNum
        #                                                    ])
        self._colorizor = colornet.CN_Colorize(outputDim=self._BinNum,
                                               name=self._name + '_Colorizor',
                                               trainable=True,
                                               bnPhase=True,
                                               reuse=False,
                                               coreActivation=self._coreAct,
                                               lastLayerActivation=self._lastAct
                                               )
        self._output = tf.nn.softmax(self._colorizor(self._inputImgs), -1)
        print "build Done!"

    def _createLoss(self):
        print "create loss..."
        weight_matrix = np.tile(np.expand_dims(np.expand_dims(self._weight_list, 0), 0), [self._imgSize[0] / 4,
                                                                                          self._imgSize[1] / 4,
                                                                                          1])
        # weight_matrix = np.tile(np.expand_dims(np.expand_dims(self._weight_list, 0), 0), [self._imgSize[0],
        #                                                                                   self._imgSize[1],
        #                                                                                   1])
        weight_matrix = tf.convert_to_tensor(weight_matrix, tf.float32)
        weight_matrix = tf.expand_dims(weight_matrix, 0)
        weight_matrix = tf.tile(weight_matrix, [tf.shape(self._outputGT)[0], 1, 1, 1])
        self._weight_matrix = weight_matrix
        indices_GT = tf.argmax(self._outputGT, -1)
        self._loss_sum_over_q = -1.0 * tf.reduce_sum(self._outputGT * tf.log(self._output + 1e-7), -1)
        # print self._loss_sum_over_q

        mask = tf.cast(tf.one_hot(indices_GT, depth=self._outputGT.get_shape()[-1]), tf.float32)
        mask = tf.reduce_sum(mask * weight_matrix, -1)
        # self._loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(mask * self._loss_sum_over_q, -1), -1), -1)
        self._loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(self._loss_sum_over_q, -1), -1), -1)

        print "create Done!"

    def _setOptimizer(self):
        print "set optimizer..."
        self._lr = tf.placeholder(tf.float32, shape=[])
        with tf.control_dependencies(self._colorizor.allUpdate_ops):
            self._optimizer = \
                tf.train.AdamOptimizer(learning_rate=self._lr).minimize(
                    self._loss,
                    None,
                    var_list=self._colorizor.allVariables)
        print "set Done!"

    def _createEvaluation(self):
        print "evaluation..."
        eval_mask = tf.cast(tf.equal(tf.argmax(self._outputGT, -1), tf.argmax(self._output, -1)), tf.float32)
        eval_mask_acc = tf.reduce_sum(tf.reduce_sum(eval_mask, -1), -1) / (self._imgSize[0] * self._imgSize[1])
        self._avg_acc = tf.reduce_mean(eval_mask_acc, -1)

        print "eval Done!"

    def fit(self, batchDict):
        feed_dict = {
            self._inputImgs: batchDict['InputImages'],
            self._outputGT: batchDict['OutputImages'],
            self._lr: batchDict['LearningRate']
        }
        opt, loss, acc\
            , w, loss_over_q\
            , outputGT, output = self._sess.run([self._optimizer, self._loss, self._avg_acc
                                                 , self._weight_matrix, self._loss_sum_over_q
                                                 , self._outputGT, self._output], feed_dict=feed_dict)

        print ("loss is {:f}".format(loss))
        print ("acc is {:f}%".format(acc * 100))

        # print outputGT.shape
        # print output.shape
        # print loss_over_q.shape

        return loss, acc

    def saveColorizorCore(self, savePath='./'):
        CorePath = os.path.join(savePath, self._name + '_colorizorCore.ckpt')
        self._colorizor.coreSaver.save(self._sess, CorePath)

    def saveColorizorLastLayer(self, savePath='./'):
        LastPath = os.path.join(savePath, self._name + '_colorizorLastLayer.ckpt')
        self._colorizor.colorizorSaver.save(self._sess, LastPath)

    def saveNetworks(self, savePath='./'):
        self.saveColorizorCore(savePath)
        self.saveColorizorLastLayer(savePath)

    def restoreColorizorCore(self, restorePath='./'):
        CorePath = os.path.join(restorePath, self._name + '_colorizorCore.ckpt')
        self._colorizor.coreSaver.restore(self._sess, CorePath)

    def restoreColorizorLastLayer(self, restorePath='./'):
        LastPath = os.path.join(restorePath, self._name + '_colorizorLastLayer.ckpt')
        self._colorizor.colorizorSaver.restore(self._sess, LastPath)

    def restoreNetworks(self, restorePath='./'):
        self.restoreColorizorCore(restorePath)
        self.restoreColorizorLastLayer(restorePath)


# sample = colornet_colorizor()

