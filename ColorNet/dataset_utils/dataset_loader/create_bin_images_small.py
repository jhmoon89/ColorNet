import cv2
import os, random, re, pickle
import numpy as np
import sys
from PIL import Image
from xml.etree.cElementTree import parse
import xml.etree.ElementTree as ET
# import dataset_utils.datasetUtils as datasetUtils
import time
# from scipy import spatial

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


print 'load data point path...'
data_path = '/ssdubuntu/data/ILSVRC'
gamut = np.load("/ssdubuntu/color/gamut.npy")
gamut_range = np.argwhere(gamut != 0)
bin_Num = len(gamut_range)
gamut_range = np.tile(np.expand_dims(np.expand_dims(gamut_range, 0), 0), [56, 56, 1, 1])
trainPath = os.path.join(data_path, 'Data')
trainPath = os.path.join(trainPath, 'VID')
trainPath = os.path.join(trainPath, 'train')
subtrainPathList = os.listdir(trainPath)
subtrainPathList.sort(key=natural_keys)

subsubtrainPathList = []
dataPointPathList = []
for subtrainpath in subtrainPathList:
    xxxx = os.path.join(trainPath, subtrainpath)
    Bin_xxxx = xxxx.replace("/ssdubuntu/data/ILSVRC/Data/VID/train", "/hdd/data/Bin_Data")
    if not os.path.exists(Bin_xxxx):
        os.makedirs(Bin_xxxx)
    sub_sub_train_path = os.listdir(xxxx)
    sub_sub_train_path.sort(key=natural_keys)
    subsubtrainPathList.append(sub_sub_train_path)
    for k in sub_sub_train_path:
        semi_finalPath = os.path.join(trainPath, subtrainpath, k)
        Bin_semi = semi_finalPath.replace("/ssdubuntu/data/ILSVRC/Data/VID/train", "/hdd/data/Bin_Data")
        if not os.path.exists(Bin_semi):
            os.makedirs(Bin_semi)
        imgPathList = os.listdir(semi_finalPath)
        imgPathList.sort(key=natural_keys)
        for img_ind in range(len(imgPathList)):
            finalPath = os.path.join(semi_finalPath, imgPathList[img_ind])
            dataPointPathList.append(finalPath)
            img = cv2.imread(finalPath)
            img = cv2.resize(img, (56, 56))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img_tile = np.tile(np.expand_dims(img[..., 1:] / 8, -2), [1, 1, bin_Num, 1])
            mask = np.equal(img_tile, gamut_range)
            mask = np.all(mask, -1).astype('int')
            mask_ind = mask.argmax(-1)
            # print mask_ind
            np.save(os.path.join(Bin_semi, imgPathList[img_ind]).replace(".JPEG", "_small.npy"), mask_ind.astype('int16'))
        print k
print 'load done!'

# a = np.load("/hdd/data/Bin_Data/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000/000000.npy")
# print a[0, 0, :]
# print dataPointPathList[1].replace(".JPEG",".npy")
