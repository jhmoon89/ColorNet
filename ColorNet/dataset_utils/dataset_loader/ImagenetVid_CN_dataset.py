import os, cv2
import numpy as np
import sys
from PIL import Image
from xml.etree.cElementTree import parse
import xml.etree.ElementTree as ET
# import dataset_utils.datasetUtils as datasetUtils
import time
# from scipy import spatial

#################################################
import dataset_utils.datasetUtils as datasetUtils

def imageResize(imagePath, imageSize, bbox):
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if bbox != None:
        imageBbox = image[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
        if len(imageBbox) == 0 or len(imageBbox[0]) == 0:
            imageResult = image
        else:
            imageResult = imageBbox
    else:
        imageResult = image
    imageResult = datasetUtils.imgAug(imageResult)
    imageResult = cv2.resize(imageResult, imageSize)
    return imageResult


class imagenetVidDataset(object):
    def __init__(self, dataPath, consecutiveLength=2, classNum=31):
        self._classes = ['__background__',  # always index 0
                         'airplane', 'antelope', 'bear', 'bicycle',
                         'bird', 'bus', 'car', 'cattle',
                         'dog', 'domestic_cat', 'elephant', 'fox',
                         'giant_panda', 'hamster', 'horse', 'lion',
                         'lizard', 'monkey', 'motorcycle', 'rabbit',
                         'red_panda', 'sheep', 'snake', 'squirrel',
                         'tiger', 'train', 'turtle', 'watercraft',
                         'whale', 'zebra']
        self._classes_map = ['__background__',  # always index 0
                             'n02691156', 'n02419796', 'n02131653', 'n02834778',
                             'n01503061', 'n02924116', 'n02958343', 'n02402425',
                             'n02084071', 'n02121808', 'n02503517', 'n02118333',
                             'n02510455', 'n02342885', 'n02374451', 'n02129165',
                             'n01674464', 'n02484322', 'n03790512', 'n02324045',
                             'n02509815', 'n02411705', 'n01726692', 'n02355227',
                             'n02129604', 'n04468005', 'n01662784', 'n04530566',
                             'n02062744', 'n02391049']
        # self._gamut = np.load("/ssdubuntu/color/gamut.npy")
        self._gamut = np.load("/ssdubuntu/color/gamut_cell16.npy")
        self._binNum = None
        self._gamut_tile = None
        self._gamut_tile_small = None
        # self._tree = None
        self._dataPath = dataPath
        self._classNum = classNum
        self._epoch = 0
        self._dataStart = 0
        self._dataLength = 0
        self._dataPointPathList = None
        self._classIdxConverter = None
        self._imageSize = (224, 224)
        self._consecutiveLength = consecutiveLength
        self._refineGamut()
        self._loadDataPointPath()
        self._dataShuffle()

    def setImageSize(self, size=(224, 224)):
        self._imageSize = (size[0], size[1])

    def _refineGamut(self):
        print 'refine gamut...'
        self._gamut_range = np.argwhere(self._gamut != 0)
        self._binNum = len(self._gamut_range)

        # gamut_tile : frame_num * H * W * binNum * 2
        self._gamut_tile = np.tile(np.expand_dims(np.expand_dims(np.expand_dims(self._gamut_range, 0), 0), 0),
                                   [self._consecutiveLength,
                                    self._imageSize[0],
                                    self._imageSize[1],
                                    1,
                                    1])
        self._gamut_tile_small = np.tile(np.expand_dims(np.expand_dims(np.expand_dims(self._gamut_range, 0), 0), 0),
                                         [self._consecutiveLength,
                                          self._imageSize[0] / 4,
                                          self._imageSize[1] / 4,
                                          1,
                                          1])
        # self._tree = spatial.KDTree(gamut_range)
        print 'refine done!'

    def index_to_onehot_bin(self, np_file):
        return (np.arange(self._binNum) == np_file[..., None]).astype('float32')

    def _loadDataPointPath(self):
        print 'load data point path...'
        self._dataPointPathList = []
        self._classIdxConverter = dict()

        # temp dataset(one video only)


        #         with open(self._dataPath + "/class.txt") as textFile:
        #             lines = [line.split(" ") for line in textFile]
        #         print lines

        trainPath = os.path.join(self._dataPath, 'Data')
        trainPath = os.path.join(trainPath, 'VID')
        trainPath = os.path.join(trainPath, 'train')
        subtrainPathList = os.listdir(trainPath)
        subtrainPathList.sort(key=datasetUtils.natural_keys)
        # print subtrainPathList
        subsubtrainPathList = []
        self._dataPointPathList = []
        for subtrainpath in subtrainPathList:
            sub_sub_train_path = os.listdir(os.path.join(trainPath, subtrainpath))
            sub_sub_train_path.sort(key=datasetUtils.natural_keys)
            subsubtrainPathList.append(sub_sub_train_path)
            for k in sub_sub_train_path:
                self._singlevideoPathList = [] # temporary
                semi_finalPath = os.path.join(trainPath, subtrainpath, k)
                imgPathList = os.listdir(semi_finalPath)
                imgPathList.sort(key=datasetUtils.natural_keys)
                if len(imgPathList) < self._consecutiveLength + 1:
                    continue
                for img in range(len(imgPathList) - self._consecutiveLength):
                    finalPath = os.path.join(semi_finalPath, imgPathList[img])
                    self._dataPointPathList.append(finalPath)
                    self._singlevideoPathList.append(finalPath)
                    # print self._dataPointPathList[-1]
                    #         print subsubtrainPathList[0]
                    #         print len(subsubtrainPathList)
                    #         print len(subsubtrainPathList[3])

        self._dataLength = len(self._dataPointPathList)
        # self._dataLength = len(self._singlevideoPathList)
        print 'load done!'

    def _dataShuffle(self):
        # 'data list shuffle...'
        self._dataStart = 0
        np.random.shuffle(self._dataPointPathList)
        print "shuffle done!\n"
        # print len(self._dataPointPathList)
        # print self._dataPointPathList[0]

    def showImage(self, path, ind):
        index = str(ind).zfill(6)
        img_name = path + '/' + index + '.JPEG'

        # print img_name
        img = cv2.imread(img_name)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def drawRect(self, img, box):
        img = cv2.rectangle(img, (box[1], box[3]), (box[0], box[2]), (255, 0, 0), 3)
        return img

    def getConsecutiveImages(self, path, startind):

        # change from BGR to Lab and get (L, ab) channels
        # img_ab_List = np.zeros([self._consecutiveLength, self._imageSize[0], self._imageSize[1], 2])
        img_List = np.zeros([self._consecutiveLength,
                             self._imageSize[0],
                             self._imageSize[1],
                             3], dtype=np.float32)
        img_List_small = np.zeros([self._consecutiveLength,
                                   self._imageSize[0] / 4,
                                   self._imageSize[1] / 4,
                                   3], dtype=np.float32)
        Q_List_small = np.zeros([self._consecutiveLength,
                                 self._imageSize[0] / 4,
                                 self._imageSize[1] / 4,
                                 self._binNum], dtype=np.float32)

        # For loop for getting concurrent images
        for i in range(self._consecutiveLength):
            # current frame
            index = str(startind + i).zfill(6)
            img_name = path + '/' + index + '.JPEG'
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img = cv2.resize(img, self._imageSize)
            img_small = cv2.resize(img, (self._imageSize[0] / 4, self._imageSize[1] / 4))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype('float32')
            img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2Lab).astype('float32')
            numpy_name = img_name.replace("/ssdubuntu/data/ILSVRC/Data/VID/train", "/ssdubuntu/Bin_Data")
            numpy_name = numpy_name.replace(".JPEG", ".npy")
            numpy_name_small = numpy_name.replace(".npy", "_small.npy")
            numpy_name_cell16 = numpy_name.replace(".npy", "_small_cell16.npy")
            numpy_small = np.load(numpy_name_cell16)
            # numpy_small = np.load(numpy_name_small)
            # numpy_org = np.load(numpy_name)

            # assign data to numpy
            img_List[i, ...] = img
            img_List_small[i, ...] = img_small
            Q_List_small[i, ...] = self.index_to_onehot_bin(numpy_small)

        return img_List, img_List_small, Q_List_small

    def newEpoch(self):
        self._epoch += 1
        self._dataStart = 0
        self._dataShuffle()

    def convert_Lab_to_bin(self, Img_List_Batch, gamut_tile, batchnum):
        gamut_tile_tile = np.tile(np.expand_dims(gamut_tile, 0), [batchnum, 1, 1, 1, 1, 1])
        Img_List_Batch_tile = np.tile(np.expand_dims(Img_List_Batch[..., 1:].astype('uint8') / 16, -2),
                                      [1, 1, 1, 1, self._binNum, 1])
        Q_List_Batch = np.equal(gamut_tile_tile, Img_List_Batch_tile)
        Q_List_Batch = np.all(Q_List_Batch, -1).astype('float32')
        L_List_Batch = Img_List_Batch[..., 0:1]
        # toc = time.time()
        # print toc-tic
        Input_Batch = np.concatenate([L_List_Batch[:, 1:, ...], Q_List_Batch[:, :-1, ...]], -1)
        Input_Batch = Input_Batch.reshape(tuple([-1]) + Input_Batch.shape[2:])
        Output_Batch = Q_List_Batch[:, 1:, ...]
        Output_Batch = Output_Batch.reshape(tuple([-1]) + Output_Batch.shape[2:])

        return Input_Batch, Output_Batch

    def getNextBatch(self, batchSize=32):
        verystartTime = time.time()
        if self._dataStart + batchSize >= self._dataLength:
            print 'new epoch'
            self.newEpoch()
        dataStart = self._dataStart
        dataEnd = dataStart + batchSize
        self._dataStart = self._dataStart + batchSize

        # print dataStart, dataEnd, self._dataLength

        # Getting Batch
        dataPathTemp = self._dataPointPathList[dataStart:dataEnd]
        # dataPathTemp = self._singlevideoPathList[dataStart:dataEnd] # temp: test for one video

        Img_List_Batch = np.zeros([batchSize,
                                  self._consecutiveLength,
                                  self._imageSize[0],
                                  self._imageSize[1],
                                  3], dtype=np.float32)
        Img_List_Batch_small = np.zeros([batchSize,
                                   self._consecutiveLength,
                                   self._imageSize[0] / 4,
                                   self._imageSize[1] / 4,
                                   3], dtype=np.float32)

        Q_List_Batch_small = np.zeros([batchSize,
                                       self._consecutiveLength,
                                       self._imageSize[0] / 4,
                                       self._imageSize[1] / 4,
                                       self._binNum], dtype=np.float32)

        for i in range(len(dataPathTemp)):
            path = dataPathTemp[i]
            parent_path = os.path.abspath(os.path.join(path, '..'))  # get the parent path
            ind = int(path.split('/')[-1].replace('.JPEG', ''))  # get the image index

            # then obtain consecutive images
            Img_List_Batch[i, ...], Img_List_Batch_small[i, ...], Q_List_Batch_small[i, ...] = \
                self.getConsecutiveImages(parent_path, ind)

        ###################################################################
        # tic = time.time()
        # gamut_tile_tile = np.tile(np.expand_dims(self._gamut_tile, 0), [len(dataPathTemp), 1, 1, 1, 1, 1])
        # Img_List_Batch_tile = np.tile(np.expand_dims(Img_List_Batch[..., 1:].astype('uint8') / 8, -2),
        #                               [1, 1, 1, 1, self._binNum, 1])
        # Q_List_Batch = np.equal(gamut_tile_tile, Img_List_Batch_tile)
        # Q_List_Batch = np.all(Q_List_Batch, -1).astype('float')
        # L_List_Batch = Img_List_Batch[..., 0:1]
        # # toc = time.time()
        # # print toc-tic
        # Input_Batch = np.concatenate([L_List_Batch[:, 1:, ...], Q_List_Batch[:, :-1, ...]], -1)
        # Input_Batch = Input_Batch.reshape(tuple([-1])+Input_Batch.shape[2:])
        # Output_Batch = Q_List_Batch[:, 1:, ...]
        # Output_Batch = Output_Batch.reshape(tuple([-1])+Output_Batch.shape[2:])
        ###################################################################
        # Input_Batch, Output_Batch = self.convert_Lab_to_bin(Img_List_Batch, self._gamut_tile, len(dataPathTemp))
        # Input_Batch_small, Output_Batch_small = self.convert_Lab_to_bin(Img_List_Batch_small,
        #                                                                 self._gamut_tile_small,
        #                                                                 len(dataPathTemp))
        ###################################################################
        # Input_Batch = Img_List_Batch[:, 1:, ..., 0:1]
        Input_Batch = np.concatenate([Img_List_Batch[:, :-1, ..., 1:], Img_List_Batch[:, 1:, ..., 0:1]], -1)
        Input_Batch = Input_Batch.reshape(tuple([-1])+Input_Batch.shape[2:])
        # _, Output_Batch_small = self.convert_Lab_to_bin(Img_List_Batch_small,
        #                                                 self._gamut_tile_small,
        #                                                 len(dataPathTemp))
        ###################################################################


        final_batchData = {
            'Paths': dataPathTemp,
            'InputImages': Input_Batch[..., -1:],
            # 'OutputImages': Q_List_Batch[:, 1:, ...].reshape(tuple([-1])+Q_List_Batch.shape[2:])
            'OutputImages': Q_List_Batch_small[:, 1:, ...].reshape(tuple([-1]) + Q_List_Batch_small.shape[2:])
            # 'OutputImages': Output_Batch_small,
            # 'OutputImages': Output_Batch,
            # 'LabImages': Img_List_Batch[:, 1:, ...].reshape(tuple([-1])+Img_List_Batch.shape[2:])
        }

        return final_batchData

#####################################################################################################
# data_path = '/ssdubuntu/data/ILSVRC'
# vid_data = imagenetVidDataset(data_path, consecutiveLength=2)
# sample = vid_data.getNextBatch(4)
# input = sample.get('InputImages')
# output = sample.get('OutputImages')
#
# print type(input[0, 0, 0, -1])
#
# print input.shape
# print output.shape
#
#
#
#
#
# for i in range(4):
#     L_channel = cv2.resize(sample.get('InputImages')[i, ..., -1].astype('uint8'), (56, 56))
#     print L_channel.shape
#     L_channel = np.expand_dims(L_channel, -1)
#     recovered_img = datasetUtils.cvt_z_to_image(output[i, ...], vid_data._gamut, 0.38)
#     print recovered_img.shape
#     recovered_img = np.concatenate([L_channel, recovered_img], -1)
#     recovered_img2 = datasetUtils.cvt_z_to_image(output[i, ...], vid_data._gamut, 0.0)
#     recovered_img2 = np.concatenate([L_channel, recovered_img2], -1)
#     recovered_img3 = datasetUtils.cvt_z_to_image(output[i, ...], vid_data._gamut, 1.0)
#     recovered_img3 = np.concatenate([L_channel, recovered_img3], -1)
#     cv2.imshow('original', cv2.cvtColor(sample.get('InputImages')[i, ..., :-1].astype('uint8'), cv2.COLOR_Lab2BGR))
#     cv2.imshow('recovered_0.38', cv2.cvtColor(recovered_img, cv2.COLOR_Lab2BGR))
#     cv2.imshow('recovered_0.0', cv2.cvtColor(recovered_img2, cv2.COLOR_Lab2BGR))
#     cv2.imshow('recovered_1.0', cv2.cvtColor(recovered_img3, cv2.COLOR_Lab2BGR))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



# gamut_tile = vid_data._gamut_tile
#
# print gamut_tile.shape
# img_name = "/ssdubuntu/data/ILSVRC/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00001003/000000.JPEG"
# img = cv2.cvtColor(cv2.resize(cv2.imread(img_name, cv2.IMREAD_COLOR), (224, 224)), cv2.COLOR_BGR2Lab)
# img = (img[..., 1:] / 8).astype('uint8')
#
# mask = np.equal(np.tile(np.expand_dims(img, -2), [1, 1, 374, 1]), gamut_tile[0])
# mask = np.logical_and(mask[..., 0], mask[..., 1]).astype('float')



# data = vid_data._cvt_image_to_z(img)



# print vid_data._dataPointPathList[:5]


