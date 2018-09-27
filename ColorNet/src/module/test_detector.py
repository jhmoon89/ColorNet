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

# import net_core.mfnet as mfnet
########################################################
import src.net_core.mfnet as mfnet


class mfnet_detector(object):
    def __init__(self,
                 dataPath='./',
                 nameScope='MF',
                 imgSize=(416, 416),
                 batchSize=32,
                 learningRate=0.0001,
                 classNum=31,
                 coreActivation=tf.nn.relu,
                 lastActivation=tf.nn.softmax,
                 concurrentFrame=4,
                 bboxNum=2
                 ):
        self._imgList = None
        self._imgClassList = None
        self._dataPath = dataPath
        self._nameScope = nameScope
        self._imgSize = imgSize
        self._batchSize = batchSize
        self._lr = learningRate
        self._coreAct = coreActivation
        self._lastAct = lastActivation
        self._classNum = classNum
        self.variables = None
        self.update_ops = None
        self._inputImgs = None
        #         self._inputImgs = tf.placeholder(tf.float32, shape=(None, 416, 416, 30))
        self._output = None
        self._outputGT = None
        self._optimizer = None
        self._loss = None
        self._concurrentFrame = concurrentFrame
        self._bboxNum = bboxNum
        self._lambdacoord = 5.0
        self._lambdaobj = 1.0
        self._lambdanoobj = 0.5
        self._lambdacls = 1.0
        self._objnum = 10  ### number of maximum object detected in a single frame
        self._indicators = None
        self._classes = None
        self._detTh = 0.5

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        # self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # initialize Vars

        self._buildNetwork()
        self._createEvaluation()

        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        self._sess = tf.Session()
        self._sess.run(init)


    def _buildNetwork(self):
        print "build Network..."
        #######################################################
        # tf.reset_default_graph()  ### modified for ipython!!!##
        #######################################################
        self._inputImgs = tf.placeholder(tf.float32, shape=(None, self._imgSize[0],
                                                            self._imgSize[1], 3 * self._concurrentFrame))
        outputDim = self._bboxNum * 5 + self._classNum
        self._outputGT = tf.placeholder(tf.float32,
                                        shape=(None, int(self._imgSize[0] / 32),
                                               int(self._imgSize[1] / 32), outputDim))

        self._detector = mfnet.MF_Detection(outputDim=outputDim,
                                            nameScope=self._nameScope + 'detector',
                                            trainable=True,
                                            bnPhase=False, ## For test, no batch normalization
                                            reuse=False,  ######## here modified for ipython(True), normally False!!
                                            coreActivation=self._coreAct,
                                            lastLayerActivation=self._lastAct
                                            )
        self._output = self._detector(self._inputImgs)

        print "build Done!"

    # def get_iou(self, box1, box2):
    #     # transform (x, y, w, h) to (x1, y1, x2, y2)
    #     box1_t = np.array([box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]])
    #     box2_t = np.array([box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]])
    #
    #     # left upper and right down
    #     lu = np.array([np.maximum(box1_t[0], box2_t[0]), np.maximum(box1_t[1], box2_t[1])])
    #     rd = np.array([np.minimum(box1_t[2], box2_t[2]), np.minimum(box1_t[3], box2_t[3])])
    #
    #     # print lu, rd
    #     # intersection
    #     intersection = np.maximum(0.0, rd - lu)
    #     # print intersection
    #     inter_area = intersection[0] * intersection[1]
    #
    #     # calculate each box area
    #     box1_area = box1[2] * box1[3]
    #     box2_area = box2[2] * box2[3]
    #
    #     union = inter_area / (box1_area + box2_area - inter_area)
    #     return union
    #
    #
    # def find_max_iou_box(self, box, box_ref_list):
    #     iou_list = []
    #     for j in range(len(box_ref_list)):
    #         iou_list.append(self.get_iou(box, box_ref_list[j]))
    #     return box_ref_list[np.argmax(iou_list)]

    def duplicate_each_element(self, T, num):  # duplicate each element 'num' times for last dimension
        temp = tf.expand_dims(T, -1)
        shape_list = [1] * len(T.get_shape()) + [num]
        temp = tf.tile(temp, shape_list)
        T_shape = T.get_shape().as_list()

        temp = tf.reshape(temp, [-1] + T_shape[1:-1] + [T_shape[-1] * num])
        return temp

    def merge_last_two_dimensions(self, T):  ## T : (None, ....)
        T_shape = T.get_shape().as_list()
        new_shape = [-1] + T_shape[1:-2] + [T_shape[-2] * T_shape[-1]]
        temp = tf.reshape(T, new_shape)
        return temp

    def split_last_dimension(self, T, last_size):  ## T : (None, ... (n * last_size) )
        T_shape = T.get_shape().as_list()
        assert T_shape[-1] % last_size == 0
        new_shape = [-1] + T_shape[1:-1] + [T_shape[-1] / last_size] + [last_size]
        temp = tf.reshape(T, new_shape)  ## output : (None, ..., n, last_size)
        return temp

    def wh_sqrt(self, boxes):  ## boxes : (None, 13, 13, bboxNum, 4)
        w = boxes[..., 2]
        h = boxes[..., 3]
        wh_sqrt = tf.stack([tf.sqrt(w), tf.sqrt(h)], -1)
        xy = boxes[..., :2]
        return tf.concat([xy, wh_sqrt], -1)

    def xy_add_offset(self, boxes):  ## boxes : (None, 13, 13, bboxNum, 4)
        x = boxes[..., 0]
        y = boxes[..., 1]

        batchSize = tf.shape(self._output)[0]
        gridSize = [int(self._imgSize[0] / 32), int(self._imgSize[1] / 32)]
        offset_col = np.transpose(np.reshape(np.array(
            [np.arange(gridSize[0])] * gridSize[1] * self._bboxNum),
            (self._bboxNum, gridSize[0], gridSize[1])), (1, 2, 0))
        offset_row = np.transpose(offset_col, (1, 0, 2))

        offset_col_tf = tf.tile(tf.reshape(tf.constant(offset_col, dtype=tf.float32),
                                           [1, gridSize[0], gridSize[1], self._bboxNum]), [batchSize, 1, 1, 1])

        offset_row_tf = tf.tile(tf.reshape(tf.constant(offset_row, dtype=tf.float32),
                                           [1, gridSize[0], gridSize[1], self._bboxNum]), [batchSize, 1, 1, 1])

        x = x + offset_col_tf
        y = y + offset_row_tf

        x = tf.expand_dims(x, -1)
        y = tf.expand_dims(y, -1)

        xy = tf.concat([x, y], -1)
        wh = boxes[..., 2:]

        return tf.concat([xy, wh], -1)

    def calc_iou(self, boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """

        # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
        boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                             boxes1[..., 1] - boxes1[..., 3] / 2.0,
                             boxes1[..., 0] + boxes1[..., 2] / 2.0,
                             boxes1[..., 1] + boxes1[..., 3] / 2.0],
                            axis=-1)

        boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                             boxes2[..., 1] - boxes2[..., 3] / 2.0,
                             boxes2[..., 0] + boxes2[..., 2] / 2.0,
                             boxes2[..., 1] + boxes2[..., 3] / 2.0],
                            axis=-1)

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
        rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[..., 0] * intersection[..., 1]

        # calculate the boxs1 square and boxs2 square
        square1 = boxes1[..., 2] * boxes1[..., 3]
        square2 = boxes2[..., 2] * boxes2[..., 3]

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def _createLoss(self):
        print "create loss..."
        ################################################################
        ## pred
        boxes_pred = self._output[..., :4 * self._bboxNum]
        conf_pred = self._output[..., 4 * self._bboxNum: 5 * self._bboxNum]
        cls_pred = self._output[..., -1 * self._classNum:]

        ## get into resonable range
        boxes_pred = tf.nn.sigmoid(boxes_pred)
        conf_pred = tf.nn.sigmoid(conf_pred)
        cls_pred = tf.nn.softmax(cls_pred)

        self._output_refined = tf.concat([boxes_pred, conf_pred, cls_pred], -1)
        # print self._output_refined.get_shape()

        ## reshape it
        boxes_pred = self.split_last_dimension(boxes_pred, 4)  ## (None, 13, 13, bboxNum, 4)

        ## get [x, y, sqrt(w), sqrt(h)]
        boxes_sqrt_pred = self.wh_sqrt(boxes_pred)  ## (None, 13, 13, bboxNum, 4)

        ## get [x + nx, y + ny, w, h]
        boxes_offset_pred = self.xy_add_offset(boxes_pred)  ## (None, 13, 13, bboxNum, 4)

        #################################################################
        ## gt
        boxes_gt = self._outputGT[..., :4 * self._bboxNum]
        conf_gt = self._outputGT[..., 4 * self._bboxNum: 5 * self._bboxNum]
        cls_gt = self._outputGT[..., -1 * self._classNum:]

        ## reshape it
        boxes_gt = self.split_last_dimension(boxes_gt, 4)  ## (None, 13, 13, bboxNum, 4)

        ## get [x, y, sqrt(w), sqrt(h)]
        boxes_sqrt_gt = self.wh_sqrt(boxes_gt)  ## (None, 13, 13, bboxNum, 4)

        ## get [x + nx, y + ny, w, h]
        boxes_offset_gt = self.xy_add_offset(boxes_gt)  ## (None, 13, 13, bboxNum, 4)

        #################################################################
        ## tile tensors

        #################################
        ## pred should be tiled as (1, 2, ..., n), (1, 2, ..., n)

        # tile prediction box
        tile_boxes_pred = tf.tile(boxes_offset_pred, [1, 1, 1, self._bboxNum, 1])
        # (None, 13, 13, bboxNum * bboxNum, 4)

        # tile prediction confidence
        tile_conf_pred = tf.tile(conf_pred, [1, 1, 1, self._bboxNum])
        # (None, 13, 13, bboxNum * bboxNum)

        #################################
        ## gt should be tiled as (1, 1, 1, ... 1), (2, 2, ..., 2)

        # tile gt box(offset)
        tile_boxes_gt = tf.tile(boxes_offset_gt, [1, 1, 1, 1, self._bboxNum])
        shape_boxes_gt = tile_boxes_gt.get_shape().as_list()
        new_shape_boxes_gt = [-1] + shape_boxes_gt[1:3] + [self._bboxNum * self._bboxNum, 4]
        tile_boxes_gt = tf.reshape(tile_boxes_gt, new_shape_boxes_gt)
        # (None, 13, 13, bboxNum * bboxNum, 4)

        # tile gt box(sqrt)
        tile_boxes_sqrt_gt = tf.tile(boxes_sqrt_gt, [1, 1, 1, 1, self._bboxNum])

        # tile gt confidence
        tile_conf_gt = self.duplicate_each_element(conf_gt, self._bboxNum)
        # (None, 13, 13, bboxNum * bboxNum)
        #################################
        #################################################################
        ## get iou for tiled boxes(offset added) and get mask

        iou = self.calc_iou(tile_boxes_pred, tile_boxes_gt)

        # split iou to get one-hot vector
        iou = self.split_last_dimension(iou, self._bboxNum)
        _, max_iou_indices = tf.nn.top_k(iou, 1)
        max_iou_indices = tf.squeeze(max_iou_indices, -1)
        mask_obj = tf.cast(tf.one_hot(max_iou_indices, depth=iou.get_shape()[-1]), tf.bool)

        mask_noobj = tf.cast(tf.logical_not(tf.reduce_any(mask_obj, -2)), tf.float32)
        mask_obj = tf.cast(self.merge_last_two_dimensions(mask_obj), tf.float32)

        #         print mask_obj.get_shape()
        #         print tile_conf_gt.get_shape()
        obj_mat = tile_conf_gt * mask_obj  ## (None, 13, 13, bboxNum * bboxNum)
        noobj_mat = mask_noobj  ## (None, 13, 13, bboxNum)
        # noobj_mat indicates the cells which are not responsible for any of gt cells

        #################################################################
        ## obtaining losses

        # box loss
        obj_mat_box = self.duplicate_each_element(obj_mat, 4)
        # (None, 13, 13, 4 * bboxNum * bboxNum)
        tile_boxes_sqrt_pred = tf.tile(boxes_sqrt_pred, [1, 1, 1, self._bboxNum, 1])
        tile_boxes_sqrt_pred = self.merge_last_two_dimensions(tile_boxes_sqrt_pred)  # (1, 2, ...)

        tile_boxes_sqrt_gt = tf.tile(boxes_sqrt_gt, [1, 1, 1, 1, self._bboxNum])
        tile_boxes_sqrt_gt = self.merge_last_two_dimensions(tile_boxes_sqrt_gt)  # (1, 1, )

        box_loss = tf.squared_difference(tile_boxes_sqrt_gt, tile_boxes_sqrt_pred) * obj_mat_box
        box_loss = tf.reduce_sum(box_loss, -1)
        box_loss = tf.reduce_sum(box_loss, -1)
        box_loss = tf.reduce_sum(box_loss, -1)
        self._box_loss = tf.reduce_mean(box_loss)

        # obj loss
        conf_obj_loss = tf.squared_difference(tile_conf_gt, tile_conf_pred) * obj_mat
        conf_obj_loss = tf.reduce_sum(conf_obj_loss, -1)
        conf_obj_loss = tf.reduce_sum(conf_obj_loss, -1)
        conf_obj_loss = tf.reduce_sum(conf_obj_loss, -1)
        self._conf_obj_loss = tf.reduce_mean(conf_obj_loss)

        # no obj loss
        conf_noobj_loss = tf.square(conf_pred) * noobj_mat  # conf_gt cell is 0
        conf_noobj_loss = tf.reduce_sum(conf_noobj_loss, -1)
        conf_noobj_loss = tf.reduce_sum(conf_noobj_loss, -1)
        conf_noobj_loss = tf.reduce_sum(conf_noobj_loss, -1)
        self._conf_noobj_loss = tf.reduce_mean(conf_noobj_loss)

        # class loss
        cls_mat = tf.cast(tf.reduce_any(tf.cast(conf_gt, tf.bool), -1), tf.float32)
        class_crossentropy = -1 * tf.reduce_sum(tf.multiply(cls_gt, tf.log(cls_pred + 1e-9)), -1)
        class_loss = class_crossentropy * cls_mat
        class_loss = tf.reduce_sum(class_loss, -1)
        class_loss = tf.reduce_sum(class_loss, -1)
        self._class_loss = tf.reduce_mean(class_loss)

        # final loss
        self._loss = self._lambdacoord * self._box_loss \
                     + self._lambdaobj * self._conf_obj_loss \
                     + self._lambdanoobj * self._conf_noobj_loss \
                     + self._lambdacls * self._class_loss



        print "create Done!"

        #################################################################

    def _createEvaluation(self):
        print "evaluation..."

        #         # print self._objMask.get_shape()
        #         # print self._objMask_dup.get_shape()

        ####################### Get Boxes List from gt and pred tensor #######################
        ## pred
        boxes_pred = self._output[..., :4 * self._bboxNum]
        conf_pred = self._output[..., 4 * self._bboxNum: 5 * self._bboxNum]
        cls_pred = self._output[..., -1 * self._classNum:]

        ## get into resonable range
        boxes_pred = tf.nn.sigmoid(boxes_pred)
        conf_pred = tf.nn.sigmoid(conf_pred)
        cls_pred = tf.nn.softmax(cls_pred)

        self._output_refined = tf.concat([boxes_pred, conf_pred, cls_pred], -1)


        class_pred = cls_pred  ### (None, 13, 13, clsNum)


        class_gt = self._outputGT[..., -1 * self._classNum:]
        conf_gt = self._outputGT[..., 4 * self._bboxNum: 5 * self._bboxNum]

        # indices_gt = tf.where(tf.equal(conf_gt, tf.ones(tf.shape(conf_gt))))
        # self._indices_gt = indices_gt
        #
        # self._boxes_list_gt = tf.gather_nd(self._boxes_gt_offset, indices_gt)
        # conf_list_gt = tf.gather_nd(conf_gt, indices_gt)
        # prob_list_gt = tf.gather_nd(class_gt, indices_gt[...,:-1])
        #
        # self._cond_prob_list_gt = tf.multiply(tf.tile(tf.expand_dims(conf_list_gt, -1), [1, self._classNum]), prob_list_gt)
        #
        # # print self._cond_prob_list_pred.get_shape()
        # # print self._cond_prob_list_gt.get_shape()


        ######################################################################################

        # get classification accuracy
        conf_bool_mask = tf.cast(tf.cast(tf.reduce_sum(conf_gt, -1), tf.bool), tf.float32)
        class_label_pred = tf.argmax(class_pred, -1)
        class_label_gt = tf.argmax(class_gt, -1)

        class_equality = tf.cast(tf.equal(class_label_gt, class_label_pred), tf.float32)
        class_equality_obj = tf.multiply(class_equality, conf_bool_mask)
        class_equality_obj = tf.reduce_sum(class_equality_obj, -1)
        class_equality_obj = tf.reduce_sum(class_equality_obj, -1)

        # print class_equality_obj.get_shape()

        obj_num_in_batch = tf.reduce_sum(conf_bool_mask, -1)
        obj_num_in_batch = tf.reduce_sum(obj_num_in_batch, -1)

        self._obj_num_in_batch = tf.reduce_sum(obj_num_in_batch)

        EachAcc = class_equality_obj / (obj_num_in_batch + 1e-9)

        self._classAcc = tf.reduce_mean(EachAcc)

        print "eval Done!"

    def eval_from_table(self, pred_list, gt_list):
        # pred_list : predicted object list. array of (x,y,w,h,conf)
        # gt_list : gt object list. array of (x,y,w,h,conf)

        TP = 0
        FP = 0
        FN = 0

        gt_num = len(gt_list)
        pred_num = len(pred_list)

        iou_mat = np.zeros([gt_num, pred_num], np.float32)

        for gt in range(gt_num):
            for pred in range(pred_num):
                iou_mat[gt, pred] = self.get_iou(gt_list[gt], pred_list[pred])
            if iou_mat[gt, :] > 0.5:
                TP = TP + 1
            else:
                FN = FN + 1

        for pred in range(pred_num):
            if iou_mat[:, pred] <= 0.5:
                FP = FP + 1

        return [TP, FP, FN]

    def fit(self, batchDict):
        feed_dict = {
            self._inputImgs: batchDict['Images'],
            self._outputGT: batchDict['Outputs'],
        }

        # xy_gt_list = self._sess.run([self._xy_gt_list], feed_dict=feed_dict)
        # print xy_gt_list

        output, classAcc, objNum, h55 = self._sess.run(
            [
                self._output_refined, self._classAcc, self._obj_num_in_batch
                , self._detector._mfnet_core.h55
            ],
            feed_dict=feed_dict)

        # box_loss = self._sess.run([self._box_loss], feed_dict=feed_dict)

        # print (output_gt[np.argwhere(output_gt==1)[0],0:3])

        # print ("classification accuracy is {:f}%".format(100 * classAcc))
        print ("obj Num is {:d}".format(int(objNum)))

        return output, classAcc, h55

    def saveDetectorCore(self, savePath='./'):
        CorePath = os.path.join(savePath, self._nameScope + '_detectorCore.ckpt')
        self._detector.coreSaver.save(self._sess, CorePath)

    def saveDetectorLastLayer(self, savePath='./'):
        LastPath = os.path.join(savePath, self._nameScope + '_detectorLastLayer.ckpt')
        self._detector.detectorSaver.save(self._sess, LastPath)

    def saveNetworks(self, savePath='./'):
        self.saveDetectorCore(savePath)
        self.saveDetectorLastLayer(savePath)

    def restoreDetectorCore(self, restorePath='./'):
        CorePath = os.path.join(restorePath, self._nameScope + '_detectorCore.ckpt')
        self._detector.coreSaver.restore(self._sess, CorePath)

    def restoreDetectorLastLayer(self, restorePath='./'):
        LastPath = os.path.join(restorePath, self._nameScope + '_detectorLastLayer.ckpt')
        self._detector.detectorSaver.restore(self._sess, LastPath)

    def restoreNetworks(self, restorePath='./'):
        self.restoreDetectorCore(restorePath)
        self.restoreDetectorLastLayer(restorePath)

    def saveSecondCore(self, savePath='./'):
        SecondPath = os.path.join(savePath, self._nameScope + '_detectorCore2.ckpt')
        self._detector.secondSaver.save(self._sess, SecondPath)

    def restoreSecondCore(self, restorePath='./'):
        SecondPath = os.path.join(restorePath, self._nameScope + '_detectorCore2.ckpt')
        self._detector.secondSaver.restore(self._sess, SecondPath)

# sample = mfnet_detector()

# mfnet_detector._sess.close()
